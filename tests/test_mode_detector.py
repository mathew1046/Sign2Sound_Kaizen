"""Tests for alphabet vs word mode detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusion.mode_detector import ModeDetector, ModeDetectorConfig  # noqa: E402


def _hands_with_motion(prev: np.ndarray | None, delta: float) -> np.ndarray:
    hands = np.zeros((42, 2), dtype=np.float32)
    hands[:10, 0] = 0.5
    if prev is not None:
        hands = prev + delta
    return hands


def _body_with_motion(prev: np.ndarray | None, delta: float) -> np.ndarray:
    body = np.zeros((33, 2), dtype=np.float32)
    body[:10, 0] = 0.5
    if prev is not None:
        body = prev + delta
    return body


class TestModeDetector:
    def test_body_motion_switches_to_word(self):
        det = ModeDetector(ModeDetectorConfig(switch_frames=3, body_word_min=0.01))
        hands = _hands_with_motion(None, 0.0)
        body = _body_with_motion(None, 0.0)
        for i in range(5):
            hands = _hands_with_motion(hands, 0.02)
            body = _body_with_motion(body, 0.03)
            det.update(hands, body)
        assert det.mode == "word"

    def test_hand_only_motion_switches_to_alphabet(self):
        det = ModeDetector(
            ModeDetectorConfig(
                switch_frames=3,
                body_spell_max=0.008,
                hand_motion_min=0.002,
                hand_body_ratio_min=2.0,
            )
        )
        hands = _hands_with_motion(None, 0.0)
        body = _body_with_motion(None, 0.0)
        for _ in range(5):
            hands = _hands_with_motion(hands, 0.015)
            body = _body_with_motion(body, 0.001)
            det.update(hands, body)
        assert det.mode == "alphabet"

    def test_hysteresis_requires_multiple_frames(self):
        det = ModeDetector(ModeDetectorConfig(switch_frames=5))
        hands = _hands_with_motion(None, 0.0)
        body = _body_with_motion(None, 0.0)
        for _ in range(3):
            hands = _hands_with_motion(hands, 0.015)
            body = _body_with_motion(body, 0.001)
            det.update(hands, body)
        assert det.mode == "word"
        for _ in range(3):
            hands = _hands_with_motion(hands, 0.015)
            det.update(hands, body)
        assert det.mode == "alphabet"

    def test_low_hand_motion_switches_back_to_word(self):
        det = ModeDetector(
            ModeDetectorConfig(
                switch_frames=3,
                body_spell_max=0.008,
                hand_motion_min=0.002,
                hand_body_ratio_min=2.0,
            )
        )
        # 1. Transition to alphabet mode first
        hands = _hands_with_motion(None, 0.0)
        body = _body_with_motion(None, 0.0)
        for _ in range(5):
            hands = _hands_with_motion(hands, 0.015)
            body = _body_with_motion(body, 0.001)
            det.update(hands, body)
        assert det.mode == "alphabet"

        # 2. Stop hand motion (rest/very low delta)
        for _ in range(15):
            hands = _hands_with_motion(hands, 0.0001)
            body = _body_with_motion(body, 0.0001)
            det.update(hands, body)
        assert det.mode == "word"

    def test_hidden_hands_switches_back_to_word(self):
        det = ModeDetector(
            ModeDetectorConfig(
                switch_frames=3,
                body_spell_max=0.008,
                hand_motion_min=0.002,
                hand_body_ratio_min=2.0,
            )
        )
        # 1. Transition to alphabet mode first
        hands = _hands_with_motion(None, 0.0)
        body = _body_with_motion(None, 0.0)
        for _ in range(5):
            hands = _hands_with_motion(hands, 0.015)
            body = _body_with_motion(body, 0.001)
            det.update(hands, body)
        assert det.mode == "alphabet"

        # 2. Make hands invisible (zeros)
        invisible_hands = np.zeros((42, 2), dtype=np.float32)
        for _ in range(5):
            det.update(invisible_hands, body)
        assert det.mode == "word"
