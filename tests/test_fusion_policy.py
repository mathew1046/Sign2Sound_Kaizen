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
        d = policy.on_mspt("hello", 0.5, 100.0)
        assert d.action == "accept_mspt"
        assert d.gloss == "hello"

    def test_rejects_uncertain(self, policy: FusionPolicy):
        d = policy.on_mspt("uncertain", 0.05, 100.0)
        assert d.action == "ignore"

    def test_glove_agreement_logged(self, policy: FusionPolicy):
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
    def test_appends_letter(self, policy: FusionPolicy):
        d = policy.on_alphabet("A", 0.9, 100.0)
        assert d.action == "append_letter"
        assert policy.spell_display == "A"

    def test_low_confidence_ignored(self, policy: FusionPolicy):
        d = policy.on_alphabet("B", 0.5, 100.0)
        assert d.action == "ignore"

    def test_flush_spell_buffer(self, policy: FusionPolicy):
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
