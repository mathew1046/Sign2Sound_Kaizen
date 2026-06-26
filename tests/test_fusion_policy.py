"""Tests for fusion policy decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusion.policy import FusionPolicy  # noqa: E402
from fusion.tokens import SignToken  # noqa: E402
from fusion.vocabulary import FusionVocabulary  # noqa: E402


@pytest.fixture
def policy() -> FusionPolicy:
    return FusionPolicy(vocab=FusionVocabulary())


def _glove_token(label: str, now: float, **meta) -> SignToken:
    defaults = {"margin": 0.4, "flex_std": 0.05, "consecutive": 5}
    defaults.update(meta)
    return SignToken(
        gloss=label,
        source="glove",
        confidence=0.9,
        timestamp=now,
        meta=defaults,
    )


class TestMsptPolicy:
    def test_accepts_confident_mspt(self, policy: FusionPolicy):
        policy.set_auto_mode("word")
        d = policy.on_mspt("hello", 0.5, 100.0)
        assert d.action == "accept_mspt"
        assert d.gloss == "hello"

    def test_rejects_uncertain(self, policy: FusionPolicy):
        d = policy.on_mspt("uncertain", 0.05, 100.0)
        assert d.action == "ignore"

    def test_blocked_in_alphabet_mode(self, policy: FusionPolicy):
        policy.soft_mode = False
        policy.set_spell_mode(True)
        d = policy.on_mspt("hello", 0.5, 100.0)
        assert d.reason == "mspt_blocked_alphabet_mode"

    def test_soft_blocked_in_alphabet_mode(self, policy: FusionPolicy):
        policy.soft_mode = True
        policy.set_spell_mode(True)
        policy.set_alphabet_weight(1.0)
        d = policy.on_mspt("hello", 0.5, 100.0)
        assert d.reason == "mspt_soft_blocked_alphabet_mode"

    def test_soft_breakthrough_in_alphabet_mode(self, policy: FusionPolicy):
        policy.soft_mode = True
        policy.set_auto_mode("alphabet")
        policy.set_alphabet_weight(0.1)  # low alphabet weight
        d = policy.on_mspt("hello", 0.95, 100.0)  # high confidence mspt
        assert d.action == "accept_mspt"

    def test_glove_agreement_logged(self, policy: FusionPolicy):
        policy.set_auto_mode("word")
        policy.on_glove(_glove_token("hello", 99.5))
        d = policy.on_mspt("hello", 0.5, 100.0)
        assert d.action == "accept_mspt"
        assert d.meta.get("glove_agreement") is True


class TestGlovePolicy:
    def test_glove_alone_rejected(self, policy: FusionPolicy):
        d = policy.on_glove(_glove_token("hello", 200.0))
        assert d.action == "ignore"
        assert d.reason == "glove_confirm_only"

    def test_glove_rest_rejected(self, policy: FusionPolicy):
        d = policy.on_glove(_glove_token("rest", 200.0))
        assert d.reason == "glove_rest"

    def test_glove_low_margin_rejected(self, policy: FusionPolicy):
        d = policy.on_glove(_glove_token("hello", 200.0, margin=0.05))
        assert d.reason == "glove_low_margin"

    def test_glove_fallback_when_mspt_silent(self):
        p = FusionPolicy(vocab=FusionVocabulary(), glove_fallback=True, glove_fallback_silent_sec=2.0)
        d = p.on_glove(_glove_token("hello", 500.0))
        assert d.action == "accept_glove"
        assert d.gloss == "hello"

    def test_glove_letter_ignored_in_spell_mode(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        d = policy.on_glove(_glove_token("b", 100.0))
        assert d.reason == "glove_letter_in_spell_mode"


class TestAlphabetPolicy:
    def test_appends_letter_in_alphabet_mode(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        d = policy.on_alphabet("A", 0.9, 100.0)
        assert d.action == "append_letter"
        assert policy.spell_display == "A"

    def test_blocked_in_word_mode(self, policy: FusionPolicy):
        policy.soft_mode = False
        policy.set_auto_mode("word")
        d = policy.on_alphabet("A", 0.9, 100.0)
        assert d.reason == "alphabet_blocked_word_mode"

    def test_soft_blocked_in_word_mode(self, policy: FusionPolicy):
        policy.soft_mode = True
        policy.set_auto_mode("word")
        policy.set_alphabet_weight(0.0)
        d = policy.on_alphabet("A", 0.9, 100.0)
        assert d.reason == "alphabet_soft_blocked_word_mode"

    def test_low_confidence_ignored(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        d = policy.on_alphabet("B", 0.5, 100.0)
        assert d.action == "ignore"

    def test_flush_spell_buffer(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        policy.on_alphabet("H", 0.9, 100.0)
        policy.on_alphabet("I", 0.9, 101.0)
        d = policy.flush_spell_buffer()
        assert d.action == "flush_spell"
        assert d.gloss == "HI"
        assert policy.spell_display == ""

    def test_glove_i_word_mode_maps(self, policy: FusionPolicy):
        p = FusionPolicy(vocab=FusionVocabulary(), glove_fallback=True, glove_fallback_silent_sec=1.0)
        d = p.on_glove(_glove_token("i", 300.0))
        assert d.action == "accept_glove"
        assert d.gloss == "i"


class TestLetterDebounce:
    def test_same_letter_within_refractory_ignored(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        d1 = policy.on_alphabet("A", 0.9, 100.0)
        assert d1.action == "append_letter"
        # Same letter 0.2s later — should be suppressed
        d2 = policy.on_alphabet("A", 0.9, 100.2)
        assert d2.action == "ignore"
        assert d2.reason == "alphabet_duplicate_letter"

    def test_same_letter_after_refractory_accepted(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        policy.on_alphabet("A", 0.9, 100.0)
        # Same letter 0.6s later — past the 0.5s refractory
        d2 = policy.on_alphabet("A", 0.9, 100.6)
        assert d2.action == "append_letter"
        assert policy.spell_display == "AA"

    def test_different_letter_not_debounced(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        policy.on_alphabet("A", 0.9, 100.0)
        # Different letter immediately after — should pass
        d2 = policy.on_alphabet("B", 0.9, 100.05)
        assert d2.action == "append_letter"
        assert policy.spell_display == "AB"

    def test_flush_resets_debounce(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        policy.on_alphabet("A", 0.9, 100.0)
        policy.flush_spell_buffer()
        # After flush, same letter should be accepted again
        d = policy.on_alphabet("A", 0.9, 100.1)
        assert d.action == "append_letter"
        assert policy.spell_display == "A"


class TestDisagreementPenalty:
    def test_disagreement_rejects_low_confidence(self):
        """When glove disagrees and calibrated confidence drops below threshold, MSPT is rejected."""
        p = FusionPolicy(vocab=FusionVocabulary(), min_mspt_confidence=0.15)
        p.set_auto_mode("word")
        # Glove predicts "hello" with good margin
        p.on_glove(_glove_token("hello", 100.0))
        # MSPT predicts "house" with low-ish confidence — disagreement should reject
        d = p.on_mspt("house", 0.16, 100.5)
        assert d.action == "ignore"
        assert d.reason == "mspt_glove_disagreement"

    def test_disagreement_still_accepts_high_confidence(self):
        """When glove disagrees but MSPT confidence is high enough, MSPT is still accepted."""
        p = FusionPolicy(vocab=FusionVocabulary(), min_mspt_confidence=0.12)
        p.set_auto_mode("word")
        p.on_glove(_glove_token("hello", 100.0))
        # MSPT predicts "house" with high confidence — penalty reduces but not below threshold
        d = p.on_mspt("house", 0.5, 100.5)
        assert d.action == "accept_mspt"
        assert d.meta.get("glove_agreement") is False


class TestSpellFlushResets:
    def test_flush_clears_letter_state(self, policy: FusionPolicy):
        policy.set_spell_mode(True)
        policy.on_alphabet("H", 0.9, 100.0)
        policy.on_alphabet("I", 0.9, 101.0)
        assert policy.spell_display == "HI"
        d = policy.flush_spell_buffer()
        assert d.action == "flush_spell"
        assert d.gloss == "HI"
        # After flush, letter state is reset
        assert policy.spell_display == ""
        # Same letter as last one should now be accepted
        d2 = policy.on_alphabet("I", 0.9, 101.5)
        assert d2.action == "append_letter"
