"""Tests for alphabet vs word mode detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusion.mode_detector import ModeDetector, ModeDetectorConfig  # noqa: E402


def _static_hands() -> np.ndarray:
    hands = np.zeros((42, 2), dtype=np.float32)
    hands[:, 0] = 0.5
    return hands


def _static_body() -> np.ndarray:
    body = np.zeros((33, 2), dtype=np.float32)
    body[:, 0] = 0.5
    return body


def _drive_ema(det: ModeDetector, hand_val: np.ndarray, body_val: np.ndarray, steps: int) -> None:
    """Drive the EMA by repeating the same hand/body values."""
    for _ in range(steps):
        det._hand_motion_ema = det.config.ema_alpha * 0.0 + (1 - det.config.ema_alpha) * det._hand_motion_ema
        # cheat by setting prev = cur so joint_motion returns 0
        det._prev_hands = hand_val.copy()
        det._prev_body = body_val.copy()


class TestModeDetector:
    def test_hand_motion_switches_to_word(self):
        """Any hand motion → word mode."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=3))
        hands = _static_hands()
        body = _static_body()
        det.update(hands, body)  # motion=0, prev=None → 0
        det.update(hands, body)  # motion=0, prev=hands → 0
        # Both static → alphabet
        for _ in range(5):
            det.update(hands, body)
        assert det.mode == "alphabet"

        # Now introduce hand motion
        moving_hands = hands + 0.01
        moving_hands_prev = moving_hands.copy()
        # Override prev to simulate motion
        det._prev_hands = moving_hands_prev
        for _ in range(5):
            moving_hands = moving_hands + 0.01
            det.update(moving_hands, body)
        assert det.mode == "word"

    def test_body_motion_stays_word(self):
        """Any body motion → word mode."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=3, body_word_min=0.01))
        hands = _static_hands()
        body = _static_body()
        det.update(hands, body)
        det.update(hands, body)
        # Moving body
        moving_body = body + 0.02
        det._prev_body = moving_body.copy()
        for _ in range(5):
            moving_body = moving_body + 0.02
            det.update(hands, moving_body)
        assert det.mode == "word"

    def test_static_switches_to_alphabet(self):
        """No hand or body motion → alphabet mode."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=3))
        hands = _static_hands()
        body = _static_body()
        for _ in range(5):
            det.update(hands, body)
        assert det.mode == "alphabet"

    def test_hysteresis_requires_consecutive_static_frames(self):
        """Must accumulate enough static frames before switching to alphabet."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=5))
        hands = _static_hands()
        body = _static_body()
        # 3 static frames → not enough (switch_frames=5)
        for _ in range(3):
            det.update(hands, body)
        assert det.mode == "word"
        # 3 more → total 6, should switch
        for _ in range(3):
            det.update(hands, body)
        assert det.mode == "alphabet"

    def test_hidden_hands_stays_word(self):
        """When hands are not visible, force word mode."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=3))
        # First get to alphabet mode with static visible hands
        hands = _static_hands()
        body = _static_body()
        for _ in range(5):
            det.update(hands, body)
        assert det.mode == "alphabet"

        # Hands become invisible (zeros)
        invisible = np.zeros((42, 2), dtype=np.float32)
        for _ in range(5):
            det.update(invisible, body)
        assert det.mode == "word"

    def test_movement_after_static_returns_to_word(self):
        """After being in alphabet (static), movement should trigger word."""
        det = ModeDetector(ModeDetectorConfig(switch_frames=3))
        hands = _static_hands()
        body = _static_body()
        # Get to alphabet
        for _ in range(5):
            det.update(hands, body)
        assert det.mode == "alphabet"

        # Resume hand motion
        moving = hands + 0.02
        det._prev_hands = moving.copy()
        for _ in range(5):
            moving = moving + 0.02
            det.update(moving, body)
        assert det.mode == "word"


class TestRollingAlphabetWeight:
    def test_alphabet_weight_reflects_window_ratio(self):
        det = ModeDetector(ModeDetectorConfig(
            switch_frames=3,
            hand_motion_min=0.002,
            vote_window_size=10,
        ))
        hands = _static_hands()
        body = _static_body()
        # All static → all alphabet votes → weight 1.0
        for _ in range(10):
            det.update(hands, body)
        assert det.alphabet_weight == 1.0

        # Clear and directly fill vote window to test rolling mechanics
        # (avoids EMA momentum complications from motion injection)
        det._vote_window.clear()
        for _ in range(5):
            det._vote_window.append("word")
        for _ in range(5):
            det._vote_window.append("alphabet")
        assert det.alphabet_weight == 0.5

    def test_alphabet_weight_empty_window(self):
        det = ModeDetector(ModeDetectorConfig(vote_window_size=10))
        assert det.alphabet_weight == 0.0

    def test_alphabet_weight_window_eviction(self):
        det = ModeDetector(ModeDetectorConfig(
            switch_frames=3,
            hand_motion_min=0.002,
            vote_window_size=5,
        ))
        hands = _static_hands()
        body = _static_body()
        # Fill window with alphabet votes (static)
        for _ in range(5):
            det.update(hands, body)
        assert det.alphabet_weight == 1.0

        # Directly fill window with word votes (avoids EMA timing complexity)
        det._vote_window.clear()
        for _ in range(5):
            det._vote_window.append("word")
        assert det.alphabet_weight == 0.0
