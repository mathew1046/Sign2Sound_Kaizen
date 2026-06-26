"""Detect finger-spelling (alphabet) vs whole-sign (MSPT word) mode from pose motion."""

from __future__ import annotations

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
    hand_motion_ema: float = 0.0
    body_motion_ema: float = 0.0
    ema_alpha: float = 0.25


@dataclass
class ModeDetector:
    """Hand-dominant, low-body motion → alphabet; high body motion → word signs."""

    config: ModeDetectorConfig = field(default_factory=ModeDetectorConfig)
    mode: SigningMode = "word"
    _prev_hands: np.ndarray | None = None
    _prev_body: np.ndarray | None = None
    _alphabet_votes: int = 0
    _word_votes: int = 0

    @property
    def alphabet_weight(self) -> float:
        """Continuous 0.0–1.0 weight toward alphabet mode.

        Based on the ratio of alphabet votes in the recent window.
        Used by FusionPolicy for soft mode transitions so high-confidence
        predictions from the "off" mode can still break through.
        """
        total = self._alphabet_votes + self._word_votes
        if total == 0:
            return 0.0 if self.mode == "word" else 1.0
        return self._alphabet_votes / total

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
        self.config.hand_motion_ema = (1 - alpha) * self.config.hand_motion_ema + alpha * hand_m
        self.config.body_motion_ema = (1 - alpha) * self.config.body_motion_ema + alpha * body_m

        hand_e = self.config.hand_motion_ema
        body_e = self.config.body_motion_ema
        hands_ok = self._hands_visible(hands)

        vote: SigningMode = self.mode
        if not hands_ok:
            vote = "word"
        elif body_e >= self.config.body_word_min:
            vote = "word"
        elif hand_e < self.config.hand_motion_min * 0.5:
            # Low/resting hand motion returns to word mode
            vote = "word"
        elif (
            hands_ok
            and body_e <= self.config.body_spell_max
            and hand_e >= self.config.hand_motion_min
            and hand_e / (body_e + 1e-6) >= self.config.hand_body_ratio_min
        ):
            vote = "alphabet"

        if vote == "alphabet":
            self._alphabet_votes += 1
            self._word_votes = 0
        elif vote == "word":
            self._word_votes += 1
            self._alphabet_votes = 0

        if self._alphabet_votes >= self.config.switch_frames:
            self.mode = "alphabet"
        elif self._word_votes >= self.config.switch_frames:
            self.mode = "word"

        return self.mode

    @property
    def debug(self) -> dict:
        return {
            "mode": self.mode,
            "alphabet_weight": self.alphabet_weight,
            "hand_motion_ema": self.config.hand_motion_ema,
            "body_motion_ema": self.config.body_motion_ema,
            "alphabet_votes": self._alphabet_votes,
            "word_votes": self._word_votes,
        }
