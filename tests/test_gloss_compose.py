"""Tests for ISL gloss → English composition."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mspt.gloss_compose import (
    ComposeResult,
    GlossComposer,
    compose_glosses,
    compose_rules,
    display_gloss,
    fallback_join,
)


class TestEnvLoading:
    def test_env_status_paths(self):
        from mspt.gloss_compose import DEFAULT_ENV_PATH, REPO_ROOT, env_status, load_project_env

        load_project_env()
        status = env_status()
        assert status["repo_root"] == str(REPO_ROOT)
        assert status["env_path"] == str(DEFAULT_ENV_PATH)
        assert status["dotenv_loaded"] is True
        assert "gemini_key_set" in status


class TestDisplayGloss:
    def test_compound(self):
        assert display_gloss("good_morning") == "Good Morning"

    def test_atomic(self):
        assert display_gloss("happy") == "Happy"


class TestComposeRules:
    def test_good_morning(self):
        result = compose_rules(["good_morning"])
        assert result.speak == "Good morning."
        assert result.source == "rules"
        assert not result.needs_llm

    def test_thank_you(self):
        result = compose_rules(["thank_you"])
        assert result.speak == "Thank you."
        assert not result.needs_llm

    def test_hello(self):
        result = compose_rules(["hello"])
        assert result.speak == "Hello."
        assert not result.needs_llm

    def test_i_happy(self):
        result = compose_rules(["i", "happy"])
        assert result.speak == "I am happy."
        assert result.changed
        assert not result.needs_llm

    def test_multi_needs_llm(self):
        result = compose_rules(["i", "want", "eat", "apple", "now"])
        assert result.needs_llm
        assert result.speak == ""


class TestFallbackJoin:
    def test_joins_glosses(self):
        result = fallback_join(["i", "go", "bank"])
        assert result.speak == "I go bank."
        assert result.source == "fallback_join"


class TestComposeGlosses:
    def test_rules_path(self):
        result = compose_glosses(["i", "happy"], use_gemini=False)
        assert result.speak == "I am happy."
        assert result.source == "rules"

    def test_fallback_when_no_gemini(self):
        result = compose_glosses(["i", "want", "eat", "apple", "now"], use_gemini=False)
        assert result.source == "fallback_join"
        assert "I" in result.speak

    @patch("mspt.gloss_compose.compose_with_gemini")
    def test_gemini_when_rules_insufficient(self, mock_gemini):
        mock_gemini.return_value = ComposeResult(
            speak="I want to eat an apple now.",
            source="gemini",
            changed=True,
        )
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            result = compose_glosses(["i", "want", "eat", "apple", "now"], use_gemini=True)
        assert result.speak == "I want to eat an apple now."
        assert result.source == "gemini"
        mock_gemini.assert_called_once()


class TestGlossComposer:
    def test_buffers_glosses(self):
        received: list[ComposeResult] = []

        def on_composed(result: ComposeResult) -> None:
            received.append(result)

        composer = GlossComposer(
            utterance_pause_sec=10.0,
            max_buffer=8,
            use_gemini=False,
            on_composed=on_composed,
        )
        t0 = time.monotonic()
        composer.add_gloss("i", 0.9, t0)
        composer.add_gloss("happy", 0.85, t0 + 1.0)
        assert composer.pending_display == "I · Happy"
        assert len(composer.buffer) == 2
        composer.close()

    def test_flush_on_max_buffer(self):
        received: list[str] = []

        def on_composed(result: ComposeResult) -> None:
            received.append(result.speak)

        composer = GlossComposer(
            utterance_pause_sec=60.0,
            max_buffer=2,
            use_gemini=False,
            on_composed=on_composed,
        )
        t0 = time.monotonic()
        composer.add_gloss("i", 0.9, t0)
        composer.add_gloss("happy", 0.9, t0 + 0.5)
        # flush is async; wait briefly for worker
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.05)
        composer.close()
        assert received == ["I am happy."]

    def test_tick_flushes_after_pause(self):
        received: list[str] = []

        def on_composed(result: ComposeResult) -> None:
            received.append(result.speak)

        composer = GlossComposer(
            utterance_pause_sec=0.2,
            max_buffer=8,
            use_gemini=False,
            on_composed=on_composed,
        )
        t0 = time.monotonic()
        composer.add_gloss("thank_you", 0.95, t0)
        composer.tick(t0 + 0.05)
        assert not received
        composer.tick(t0 + 0.25)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.05)
        composer.close()
        assert received == ["Thank you."]

    def test_skips_uncertain(self):
        composer = GlossComposer(use_gemini=False, on_composed=lambda r: None)
        composer.add_gloss("uncertain", 0.1, time.monotonic())
        assert composer.buffer == []
        composer.close()

    def test_manual_flush(self):
        received: list[str] = []

        def on_composed(result: ComposeResult) -> None:
            received.append(result.speak)

        composer = GlossComposer(
            utterance_pause_sec=60.0,
            max_buffer=8,
            use_gemini=False,
            on_composed=on_composed,
        )
        t0 = time.monotonic()
        composer.add_gloss("good_morning", 0.9, t0)
        composer.flush()
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.05)
        composer.close()
        assert received == ["Good morning."]
