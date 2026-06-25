"""Tests for orientation feature extraction and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dashboard.config import ORIENTATION_REFS_DIR, PROJECT_ROOT, WHOLEBODY_DIR
from dashboard.server.orientation.compare import compare_sequences
from dashboard.server.orientation.features import extract_sequence_features, palm_normal_angle_deg
from dashboard.server.orientation.reference import get_orientation_reference


@pytest.fixture
def bank_wholebody():
    path = WHOLEBODY_DIR / "bank" / "MVI_3335.npy"
    if not path.is_file():
        pytest.skip("wholebody cache not present")
    return np.load(path)


def test_extract_features_shape(bank_wholebody):
    feats = extract_sequence_features(bank_wholebody, fps=25.0)
    assert len(feats) == bank_wholebody.shape[0]
    f0 = feats[0]
    assert len(f0["palm_normal"]) == 3
    assert "index" in f0["finger_curls"]
    assert 0 <= f0["confidence"] <= 1.0


def test_self_comparison_low_error(bank_wholebody):
    ref_path = ORIENTATION_REFS_DIR / "bank.json"
    if not ref_path.is_file():
        pytest.skip("orientation refs not built")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    user_feats = extract_sequence_features(bank_wholebody, fps=25.0, active_hand=reference.get("active_hand"))
    result = compare_sequences("bank", user_feats, reference, fps=25.0)
    assert result.overall_result in ("pass", "needs_correction")
    assert result.usable_frame_ratio >= 0.6


def test_palm_normal_angle():
    a = [0.0, 0.0, 1.0]
    b = [0.0, 0.0, 1.0]
    assert palm_normal_angle_deg(a, b) == pytest.approx(0.0, abs=0.01)


def test_reference_loader():
    ref = get_orientation_reference("bank")
    if ref is None:
        pytest.skip("orientation refs not built")
    assert ref["sign_id"] == "bank"
    assert len(ref["reference_sequence"]) > 0
