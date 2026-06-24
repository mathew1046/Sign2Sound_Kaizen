"""Frame-diff motion detection for autonomous clip boundaries."""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Detect motion via grayscale frame difference."""

    def __init__(self, threshold: float = 0.008, downscale: int = 4):
        self.threshold = threshold
        self.downscale = downscale
        self._prev_gray: np.ndarray | None = None
        self._consecutive_motion = 0
        self._consecutive_still = 0
        self.last_motion = 0.0

    def reset(self) -> None:
        self._prev_gray = None
        self._consecutive_motion = 0
        self._consecutive_still = 0
        self.last_motion = 0.0

    def _to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            (max(1, w // self.downscale), max(1, h // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def update(self, frame: np.ndarray) -> float:
        gray = self._to_gray_small(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            self.last_motion = 0.0
            return 0.0

        diff = np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)) / 255.0
        self.last_motion = float(np.mean(diff))
        self._prev_gray = gray
        return self.last_motion

    def is_moving(self) -> bool:
        return self.last_motion >= self.threshold

    def tick_motion_start(self, start_frames: int) -> bool:
        """True once motion sustained for start_frames consecutive updates."""
        if self.is_moving():
            self._consecutive_motion += 1
            self._consecutive_still = 0
        else:
            self._consecutive_motion = 0
        return self._consecutive_motion >= start_frames

    def tick_stillness(self, still_frames: int) -> bool:
        """True once still for still_frames consecutive updates."""
        if not self.is_moving():
            self._consecutive_still += 1
            self._consecutive_motion = 0
        else:
            self._consecutive_still = 0
        return self._consecutive_still >= still_frames

    def reset_counters(self) -> None:
        self._consecutive_motion = 0
        self._consecutive_still = 0
