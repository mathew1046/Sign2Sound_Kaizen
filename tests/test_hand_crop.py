"""Tests for alphabet hand crop utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "alphabet_transformer"))

from alphabet_transformer.hand_crop import (  # noqa: E402
    remap_landmarks_to_full_frame,
    union_hand_bbox,
)


class TestUnionHandBbox:
    def test_empty_returns_none(self):
        assert union_hand_bbox([], 640, 480) is None

    def test_single_detection(self):
        dets = [(0.9, 100, 50, 80, 120)]
        bbox = union_hand_bbox(dets, 640, 480, pad_frac=0.0)
        assert bbox == (100, 50, 80, 120)

    def test_union_two_hands(self):
        dets = [
            (0.9, 50, 100, 60, 80),
            (0.85, 200, 110, 70, 90),
        ]
        bbox = union_hand_bbox(dets, 640, 480, pad_frac=0.0)
        x0, y0, w, h = bbox
        assert x0 == 50
        assert y0 == 100
        assert x0 + w == 270
        assert y0 + h == 200

    def test_clamps_to_frame_with_padding(self):
        dets = [(0.9, 0, 0, 40, 40)]
        bbox = union_hand_bbox(dets, 100, 100, pad_frac=0.5)
        x0, y0, w, h = bbox
        assert x0 == 0
        assert y0 == 0
        assert x0 + w <= 100
        assert y0 + h <= 100


class TestRemapLandmarks:
    def test_identity_full_frame(self):
        landmarks = np.array([[0.5, 0.5, 0.1]], dtype=np.float32)
        out = remap_landmarks_to_full_frame(landmarks, (0, 0, 640, 480), (640, 480))
        np.testing.assert_allclose(out[0], [0.5, 0.5, 0.1], rtol=1e-5)

    def test_crop_corner_remaps_to_full_frame(self):
        # Top-left of crop maps into full frame
        landmarks = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        crop = (100, 100, 200, 200)
        out = remap_landmarks_to_full_frame(landmarks, crop, (400, 400))
        np.testing.assert_allclose(out[0, :2], [0.25, 0.25], rtol=1e-5)

    def test_zero_landmarks_remap_to_crop_origin(self):
        landmarks = np.zeros((42, 3), dtype=np.float32)
        out = remap_landmarks_to_full_frame(landmarks, (10, 10, 100, 100), (640, 480))
        np.testing.assert_allclose(out[0, :2], [10 / 640, 10 / 480], rtol=1e-5)
        assert out[0, 2] == 0.0


class TestFallbackPath:
    def test_union_none_when_no_detections(self):
        assert union_hand_bbox([], 320, 240) is None

    def test_tiny_bbox_rejected(self):
        dets = [(0.5, 10, 10, 5, 5)]
        assert union_hand_bbox(dets, 640, 480, min_size=12) is None
