"""Fusion decision logic for MSPT + alphabet + glove tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fusion.calibration import (
    ALPHABET_CALIBRATION,
    GLOVE_CALIBRATION,
    MSPT_CALIBRATION,
    CalibrationParams,
    disagreement_penalty,
    noisy_or,
)
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

    # --- Soft mode transition parameters ---
    soft_mode: bool = True
    soft_mode_breakthrough: float = 0.90
    # alphabet_weight is supplied externally from ModeDetector

    # --- Spell auto-flush ---
    spell_idle_timeout_sec: float = 2.0

    # --- Letter debounce ---
    letter_refractory_sec: float = 0.5
    last_letter: str = ""
    last_letter_time: float = 0.0

    # --- Calibration ---
    mspt_calibration: CalibrationParams = field(default_factory=lambda: MSPT_CALIBRATION)
    alphabet_calibration: CalibrationParams = field(default_factory=lambda: ALPHABET_CALIBRATION)
    glove_calibration: CalibrationParams = field(default_factory=lambda: GLOVE_CALIBRATION)

    auto_mode: str = "word"
    manual_mode: ManualMode = None
    _alphabet_weight: float = 0.0

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

    @property
    def alphabet_weight(self) -> float:
        """Continuous mode weight (0.0 = word, 1.0 = alphabet)."""
        if self.manual_mode is not None:
            return 1.0 if self.manual_mode == "alphabet" else 0.0
        return self._alphabet_weight

    def set_alphabet_weight(self, weight: float) -> None:
        """Update the soft mode weight from ModeDetector."""
        self._alphabet_weight = max(0.0, min(1.0, weight))

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

    # --- Internal helpers ---

    def _prune_glove(self, now: float) -> None:
        self.recent_glove = [
            t for t in self.recent_glove if now - t.timestamp <= self.glove_agree_window_sec
        ]

    def _glove_agrees(self, mspt_gloss: str, now: float) -> tuple[bool, float]:
        """Check if any recent glove token agrees with mspt_gloss.

        Returns (agreed, best_glove_calibrated_confidence).
        """
        self._prune_glove(now)
        best_conf = 0.0
        for token in self.recent_glove:
            slug = self.vocab.glove_to_mspt_slug(token.gloss)
            if slug == mspt_gloss:
                cal = self.glove_calibration.calibrate(token.confidence)
                best_conf = max(best_conf, cal)
        return best_conf > 0.0, best_conf

    def _recent_glove_label(self, now: float) -> tuple[str | None, float]:
        """Return the most recent glove label + calibrated confidence within the window."""
        self._prune_glove(now)
        if not self.recent_glove:
            return None, 0.0
        latest = self.recent_glove[-1]
        slug = self.vocab.glove_to_mspt_slug(latest.gloss)
        cal = self.glove_calibration.calibrate(latest.confidence)
        return slug, cal

    # --- Decision entry points ---

    def on_mspt(self, gloss: str, confidence: float, now: float) -> FusionDecision:
        # Apply soft mode scaling instead of hard blocking
        if self.soft_mode and self.is_alphabet_mode:
            word_weight = 1.0 - self.alphabet_weight
            scaled_conf = confidence * word_weight
            if scaled_conf < self.soft_mode_breakthrough * self.min_mspt_confidence:
                return FusionDecision("ignore", reason="mspt_soft_blocked_alphabet_mode")
            # Allow breakthrough if raw confidence is very high
            if confidence < self.soft_mode_breakthrough:
                return FusionDecision("ignore", reason="mspt_soft_blocked_alphabet_mode")
        elif not self.soft_mode and self.is_alphabet_mode:
            return FusionDecision("ignore", reason="mspt_blocked_alphabet_mode")

        if not gloss or gloss == "uncertain" or confidence < self.min_mspt_confidence:
            return FusionDecision("ignore", reason="mspt_uncertain")

        # Calibrate MSPT confidence
        cal_mspt = self.mspt_calibration.calibrate(confidence)

        # Check glove agreement and boost via Noisy-OR
        agreed, glove_cal = self._glove_agrees(gloss, now)
        if agreed and glove_cal > 0.0:
            boosted = noisy_or(cal_mspt, glove_cal)
            meta = {
                "glove_agreement": True,
                "raw_confidence": confidence,
                "calibrated": cal_mspt,
                "glove_calibrated": glove_cal,
                "boosted": boosted,
            }
            self.last_mspt_time = now
            self.last_mspt_gloss = gloss
            return FusionDecision(
                "accept_mspt",
                gloss=gloss,
                confidence=boosted,
                reason="mspt_glove_boosted",
                meta=meta,
            )

        # Check for glove disagreement penalty
        glove_label, glove_conf = self._recent_glove_label(now)
        if glove_label is not None and glove_label != gloss and glove_conf > 0.3:
            cal_mspt = disagreement_penalty(cal_mspt)
            if cal_mspt < self.mspt_calibration.calibrate(self.min_mspt_confidence):
                return FusionDecision("ignore", reason="mspt_glove_disagreement")

        self.last_mspt_time = now
        self.last_mspt_gloss = gloss
        meta = {
            "glove_agreement": False,
            "raw_confidence": confidence,
            "calibrated": cal_mspt,
        }
        return FusionDecision(
            "accept_mspt",
            gloss=gloss,
            confidence=cal_mspt,
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
                    cal = self.glove_calibration.calibrate(token.confidence)
                    return FusionDecision(
                        "accept_glove",
                        gloss=slug,
                        confidence=cal,
                        reason="glove_fallback",
                        meta={"glove_label": label, "calibrated": cal},
                    )

        return FusionDecision("ignore", reason="glove_confirm_only")

    def on_alphabet(self, letter: str, confidence: float, now: float) -> FusionDecision:
        # Soft mode: allow high-confidence alphabet through word mode
        if self.soft_mode and self.is_word_mode:
            alpha_w = self.alphabet_weight
            scaled_conf = confidence * alpha_w
            if scaled_conf < self.soft_mode_breakthrough * self.alphabet_threshold:
                return FusionDecision("ignore", reason="alphabet_soft_blocked_word_mode")
            if confidence < self.soft_mode_breakthrough:
                return FusionDecision("ignore", reason="alphabet_soft_blocked_word_mode")
        elif not self.soft_mode and self.is_word_mode:
            return FusionDecision("ignore", reason="alphabet_blocked_word_mode")

        if confidence < self.alphabet_threshold:
            return FusionDecision("ignore", reason="alphabet_low_conf")
        if not letter or len(letter) != 1:
            return FusionDecision("ignore", reason="alphabet_invalid")

        # Debounce: suppress same letter repeated within refractory window
        upper_letter = letter.upper()
        if upper_letter == self.last_letter and (now - self.last_letter_time) < self.letter_refractory_sec:
            return FusionDecision("ignore", reason="alphabet_duplicate_letter")

        cal = self.alphabet_calibration.calibrate(confidence)
        self.last_spell_activity = now
        self.spell_buffer += letter.upper()
        self.last_letter = upper_letter
        self.last_letter_time = now
        return FusionDecision(
            "append_letter",
            gloss=letter.upper(),
            confidence=cal,
            reason="alphabet_letter",
            meta={"spell_buffer": self.spell_buffer, "calibrated": cal},
        )

    def flush_spell_buffer(self) -> FusionDecision:
        if not self.spell_buffer:
            return FusionDecision("ignore", reason="spell_buffer_empty")
        word = self.spell_buffer
        self.spell_buffer = ""
        self.last_letter = ""
        self.last_letter_time = 0.0
        return FusionDecision("flush_spell", gloss=word, reason="spell_flushed")

    def check_spell_timeout(self, now: float) -> FusionDecision:
        """Auto-flush the spell buffer after idle timeout.

        Call this every frame from the main loop.
        """
        if not self.spell_buffer:
            return FusionDecision("ignore", reason="spell_buffer_empty")
        if self.last_spell_activity <= 0.0:
            return FusionDecision("ignore", reason="spell_no_activity")
        if now - self.last_spell_activity >= self.spell_idle_timeout_sec:
            return self.flush_spell_buffer()
        return FusionDecision("ignore", reason="spell_not_timed_out")

    @property
    def spell_display(self) -> str:
        return self.spell_buffer
