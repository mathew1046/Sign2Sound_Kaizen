"""Detect finger-spelling (alphabet) vs whole-sign (MSPT word) mode from pose motion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

SigningMode = str  # "word" | "alphabet"


@dataclass
class ModeDetectorConfig:
    hand_motion_min: float = 0.003
    body_spell_max: float = 0.008
    body_word_min: float = 0.014
    hand_body_ratio_min: float = 2.5
    min_visible_hand_joints: int = 8
    switch_frames: int = 6
    vote_window_size: int = 30
    ema_alpha: float = 0.25


@dataclass
class ModeDetector:
    """Hand-dominant, low-body motion → alphabet; high body motion → word signs."""

    config: ModeDetectorConfig = field(default_factory=ModeDetectorConfig)
    mode: SigningMode = "word"
    _prev_hands: np.ndarray | None = None
    _prev_body: np.ndarray | None = None
    _vote_window: deque = field(init=False)
    _hand_motion_ema: float = 0.0
    _body_motion_ema: float = 0.0

    def __post_init__(self) -> None:
        self._vote_window = deque(maxlen=self.config.vote_window_size)

    @property
    def alphabet_weight(self) -> float:
        """Continuous 0.0–1.0 weight toward alphabet mode based on rolling vote window."""
        if not self._vote_window:
            return 0.0 if self.mode == "word" else 1.0
        return sum(1 for v in self._vote_window if v == "alphabet") / len(self._vote_window)

    def _joint_motion(self, cur: np.ndarray, prev: np.ndarray | None) -> float:
        if prev is None or cur.shape != prev.shape:
            return 0.0
        cur_f = cur.reshape(-1)
        prev_f = prev.reshape(-1)
        valid = (cur_f != 0) & (prev_f != 0)
        if not valid.any():
            return 0.0
        return float(np.mean(np.abs(cur_f[valid] - prev_f[valid])))

    def _hands_visible(self, hands: np.ndarray) -> bool:
        if hands.size == 0:
            return False
        valid = (hands != 0).any(axis=-1)
        return int(valid.sum()) >= self.config.min_visible_hand_joints

    def update(self, hands: np.ndarray, body: np.ndarray) -> SigningMode:
        hand_m = self._joint_motion(hands, self._prev_hands)
        body_m = self._joint_motion(body, self._prev_body)
        self._prev_hands = hands.copy()
        self._prev_body = body.copy()

        alpha = self.config.ema_alpha
        self._hand_motion_ema = (1 - alpha) * self._hand_motion_ema + alpha * hand_m
        self._body_motion_ema = (1 - alpha) * self._body_motion_ema + alpha * body_m

        hand_e = self._hand_motion_ema
        body_e = self._body_motion_ema
        hands_ok = self._hands_visible(hands)

        # Alphabet only when entire skeleton is static.
        # Any hand or body movement → word mode (MSPT).
        vote: SigningMode = "word"
        if hands_ok and hand_e < self.config.hand_motion_min and body_e < self.config.body_word_min:
            vote = "alphabet"
        elif not hands_ok:
            vote = "word"

        self._vote_window.append(vote)

        # Count consecutive same votes from the end of the window
        consecutive = 0
        for v in reversed(self._vote_window):
            if v == vote:
                consecutive += 1
            else:
                break
        if consecutive >= self.config.switch_frames:
            self.mode = vote

        return self.mode

    @property
    def debug(self) -> dict:
        return {
            "mode": self.mode,
            "alphabet_weight": self.alphabet_weight,
            "hand_motion_ema": self._hand_motion_ema,
            "body_motion_ema": self._body_motion_ema,
            "alphabet_votes": sum(1 for v in self._vote_window if v == "alphabet"),
            "word_votes": sum(1 for v in self._vote_window if v == "word"),
        }
