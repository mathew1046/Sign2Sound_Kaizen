"""Fusion decision logic for MSPT + alphabet + glove tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fusion.tokens import SignToken
from fusion.vocabulary import FusionVocabulary

ManualMode = Literal["word", "alphabet"] | None


@dataclass
class FusionDecision:
    action: str  # accept_mspt | accept_glove | append_letter | flush_spell | ignore
    gloss: str = ""
    confidence: float = 0.0
    reason: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class FusionPolicy:
    vocab: FusionVocabulary
    min_mspt_confidence: float = 0.12
    alphabet_threshold: float = 0.85
    glove_margin_threshold: float = 0.25
    glove_activity_threshold: float = 0.02
    glove_consecutive: int = 5
    glove_agree_window_sec: float = 1.0
    glove_fallback: bool = False
    glove_fallback_silent_sec: float = 3.0

    auto_mode: str = "word"
    manual_mode: ManualMode = None

    last_mspt_time: float = 0.0
    last_mspt_gloss: str = ""
    recent_glove: list[SignToken] = field(default_factory=list)
    spell_buffer: str = ""
    last_spell_activity: float = 0.0

    @property
    def is_alphabet_mode(self) -> bool:
        if self.manual_mode is not None:
            return self.manual_mode == "alphabet"
        return self.auto_mode == "alphabet"

    @property
    def is_word_mode(self) -> bool:
        return not self.is_alphabet_mode

    @property
    def spell_mode(self) -> bool:
        return self.is_alphabet_mode

    def set_auto_mode(self, mode: str) -> None:
        if mode in ("word", "alphabet"):
            self.auto_mode = mode

    def toggle_manual_mode(self) -> str:
        """Cycle: auto → force alphabet → force word → auto."""
        if self.manual_mode is None:
            self.manual_mode = "alphabet"
        elif self.manual_mode == "alphabet":
            self.manual_mode = "word"
        else:
            self.manual_mode = None
        return self.mode_label

    @property
    def mode_label(self) -> str:
        if self.manual_mode == "alphabet":
            return "SPELL (manual)"
        if self.manual_mode == "word":
            return "WORD (manual)"
        return "SPELL" if self.auto_mode == "alphabet" else "WORD"

    def set_spell_mode(self, enabled: bool) -> None:
        self.manual_mode = "alphabet" if enabled else "word"

    def toggle_spell_mode(self) -> bool:
        self.toggle_manual_mode()
        return self.is_alphabet_mode

    def _prune_glove(self, now: float) -> None:
        self.recent_glove = [
            t for t in self.recent_glove if now - t.timestamp <= self.glove_agree_window_sec
        ]

    def _glove_agrees(self, mspt_gloss: str, now: float) -> bool:
        self._prune_glove(now)
        for token in self.recent_glove:
            slug = self.vocab.glove_to_mspt_slug(token.gloss)
            if slug == mspt_gloss:
                return True
        return False

    def on_mspt(self, gloss: str, confidence: float, now: float) -> FusionDecision:
        if self.is_alphabet_mode:
            return FusionDecision("ignore", reason="mspt_blocked_alphabet_mode")
        if not gloss or gloss == "uncertain" or confidence < self.min_mspt_confidence:
            return FusionDecision("ignore", reason="mspt_uncertain")

        agreed = self._glove_agrees(gloss, now)
        self.last_mspt_time = now
        self.last_mspt_gloss = gloss
        meta = {"glove_agreement": agreed}
        return FusionDecision(
            "accept_mspt",
            gloss=gloss,
            confidence=confidence,
            reason="mspt_primary",
            meta=meta,
        )

    def on_glove(self, token: SignToken) -> FusionDecision:
        label = token.gloss
        if self.vocab.should_reject(label):
            return FusionDecision("ignore", reason="glove_rest")

        margin = float(token.meta.get("margin", 0.0))
        flex_std = float(token.meta.get("flex_std", 0.0))
        consecutive = int(token.meta.get("consecutive", 0))

        if margin < self.glove_margin_threshold:
            return FusionDecision("ignore", reason="glove_low_margin")
        if flex_std < self.glove_activity_threshold:
            return FusionDecision("ignore", reason="glove_low_activity")
        if consecutive < self.glove_consecutive:
            return FusionDecision("ignore", reason="glove_not_stable")

        if self.is_alphabet_mode and self.vocab.is_glove_letter(label):
            return FusionDecision("ignore", reason="glove_letter_in_spell_mode")

        slug = self.vocab.glove_to_mspt_slug(label)
        if slug is None:
            return FusionDecision("ignore", reason="glove_unknown_label")

        self.recent_glove.append(token)
        self._prune_glove(token.timestamp)

        if slug == self.last_mspt_gloss and (token.timestamp - self.last_mspt_time) <= self.glove_agree_window_sec:
            return FusionDecision("ignore", reason="glove_confirms_mspt_already_emitted")

        if self.glove_fallback and slug is not None:
            if not (self.is_alphabet_mode and self.vocab.is_glove_letter(label)):
                silent = token.timestamp - self.last_mspt_time
                if silent >= self.glove_fallback_silent_sec:
                    return FusionDecision(
                        "accept_glove",
                        gloss=slug,
                        confidence=token.confidence,
                        reason="glove_fallback",
                        meta={"glove_label": label},
                    )

        return FusionDecision("ignore", reason="glove_confirm_only")

    def on_alphabet(self, letter: str, confidence: float, now: float) -> FusionDecision:
        if self.is_word_mode:
            return FusionDecision("ignore", reason="alphabet_blocked_word_mode")
        if confidence < self.alphabet_threshold:
            return FusionDecision("ignore", reason="alphabet_low_conf")
        if not letter or len(letter) != 1:
            return FusionDecision("ignore", reason="alphabet_invalid")

        self.last_spell_activity = now
        self.spell_buffer += letter.upper()
        return FusionDecision(
            "append_letter",
            gloss=letter.upper(),
            confidence=confidence,
            reason="alphabet_letter",
            meta={"spell_buffer": self.spell_buffer},
        )

    def flush_spell_buffer(self) -> FusionDecision:
        if not self.spell_buffer:
            return FusionDecision("ignore", reason="spell_buffer_empty")
        word = self.spell_buffer
        self.spell_buffer = ""
        return FusionDecision("flush_spell", gloss=word, reason="spell_flushed")

    @property
    def spell_display(self) -> str:
        return self.spell_buffer
