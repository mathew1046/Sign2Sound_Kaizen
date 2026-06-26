"""Confidence calibration for multi-model fusion.

Different models produce softmax outputs over vastly different numbers of classes,
making their raw confidences incomparable:
  - MSPT: 263 classes → a "good" confidence is ~0.3–0.6
  - Alphabet: 26 classes → a "good" confidence is ~0.85+
  - GloveTalk: 11 classes → a "good" confidence is ~0.7+

This module provides simple rescaling to a common [0, 1] confidence scale,
plus Noisy-OR combination for multi-modal agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalibrationParams:
    """Per-model calibration: logit-adjusted rescaling.

    The baseline is ``1 / num_classes`` (chance level). Calibrated confidence
    maps [chance, 1.0] → [0, 1] linearly, then optionally applies a temperature
    to sharpen/soften the output.
    """

    num_classes: int
    temperature: float = 1.0

    @property
    def chance(self) -> float:
        return 1.0 / max(self.num_classes, 1)

    def calibrate(self, raw_confidence: float) -> float:
        """Rescale raw softmax confidence to a chance-adjusted [0, 1] scale."""
        floor = self.chance
        if raw_confidence <= floor:
            return 0.0
        calibrated = (raw_confidence - floor) / (1.0 - floor)
        if self.temperature != 1.0:
            calibrated = calibrated ** (1.0 / self.temperature)
        return min(1.0, max(0.0, calibrated))


# Default calibration parameters for the three models.
MSPT_CALIBRATION = CalibrationParams(num_classes=263, temperature=1.0)
ALPHABET_CALIBRATION = CalibrationParams(num_classes=26, temperature=1.0)
GLOVE_CALIBRATION = CalibrationParams(num_classes=11, temperature=1.0)


def noisy_or(p1: float, p2: float) -> float:
    """Noisy-OR combination: P(at least one correct) = 1 - (1-p1)(1-p2).

    Used when two independent models agree on the same prediction.
    """
    return 1.0 - (1.0 - p1) * (1.0 - p2)


def disagreement_penalty(confidence: float, penalty: float = 0.15) -> float:
    """Reduce confidence when another modality actively disagrees."""
    return max(0.0, confidence - penalty)
