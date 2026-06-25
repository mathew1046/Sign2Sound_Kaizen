"""Tests for sign boundary segmentation."""

from __future__ import annotations

import numpy as np

from mspt.segmentation import (
    ConfidenceSegmenter,
    EnsembleSegmenter,
    FixedSegmenter,
    LearnedSegmenter,
    MotionSegmenter,
    SegmenterConfig,
    trim_buffer_by_motion,
)


def _wb_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wb = np.zeros((133, 4), dtype=np.float32)
    wb[91:133, :2] = rng.random((42, 2))
    wb[91:133, 2] = 0.9
    wb[91:133, 3] = 1.0
    wb[0:17, :2] = rng.random((17, 2)) * 0.1
    wb[0:17, 3] = 1.0
    return wb


def test_trim_buffer_by_motion():
    buf = [_wb_frame(i) for i in range(10)]
    motions = [0.0, 0.0, 0.02, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    trimmed = trim_buffer_by_motion(buf, motions, motion_threshold=0.01, pad_frames=1)
    assert 3 <= len(trimmed) <= 7
    assert trimmed[0] is buf[1] or trimmed[0] is buf[2]


def test_motion_segmenter_onset_and_end():
    cfg = SegmenterConfig(
        onset_frames=2,
        offset_frames=2,
        motion_on_threshold=0.01,
        motion_off_threshold=0.005,
        min_frames=3,
        cooldown_sec=0.1,
    )
    seg = MotionSegmenter(cfg)
    t = 0.0
    prev = None
    events = []
    for i in range(5):
        wb = _wb_frame(i)
        hands = np.concatenate([wb[91:112, :2], wb[112:133, :2]])
        body = np.zeros((33, 2), np.float32)
        body[:17] = wb[0:17, :2]
        from mspt.segmentation import frame_motion

        m = 0.02 if i >= 1 else 0.0
        events.extend(seg.update(wb, m, t))
        t += 0.1

    assert any(e.kind == "start" for e in events)

    for i in range(5, 12):
        wb = _wb_frame(i)
        m = 0.0
        events.extend(seg.update(wb, m, t))
        t += 0.1

    end_events = [e for e in events if e.kind == "end"]
    assert end_events
    assert len(end_events[0].buffer or []) >= cfg.min_frames


def test_confidence_segmenter_with_mock_predict():
    cfg = SegmenterConfig(
        stable_windows=2,
        unstable_windows=2,
        conf_on=0.2,
        conf_off=0.05,
        min_frames=4,
        confidence_window_sec=0.3,
        confidence_stride_sec=0.1,
        extract_fps=10.0,
        cooldown_sec=0.1,
    )
    call_count = {"n": 0}

    def predict(buf):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return [("hello", 0.5)]
        return [("hello", 0.02)]

    seg = ConfidenceSegmenter(cfg, predict)
    t = 0.0
    events = []
    for i in range(30):
        wb = _wb_frame(i)
        m = 0.02
        events.extend(seg.update(wb, m, t))
        t += 0.1
    assert any(e.kind == "start" for e in events)


def test_learned_segmenter_thresholds():
    from mspt.segmenter_model import SignFrameClassifier

    model = SignFrameClassifier()
    model.eval()
    cfg = SegmenterConfig(onset_frames=1, offset_frames=1, min_frames=2, prob_on=0.0, prob_off=1.0)
    seg = LearnedSegmenter(cfg, model)
    t = 0.0
    events = []
    for i in range(6):
        events.extend(seg.update(_wb_frame(i), 0.02, t))
        t += 0.1
    assert any(e.kind == "start" for e in events)


def test_fixed_segmenter_timer():
    cfg = SegmenterConfig(clip_sec=0.25, min_frames=2, min_motion_frames=1, motion_threshold=0.001)
    seg = FixedSegmenter(cfg)
    seg.start_session(0.0)
    t = 0.0
    events = []
    for i in range(10):
        wb = _wb_frame(i)
        events.extend(seg.update(wb, 0.05, t))
        t += 0.05
    assert seg.phase.value in ("hold", "gap", "recording")
    assert any(e.kind == "end" for e in events)


def test_ensemble_segmenter_motion_only():
    cfg = SegmenterConfig(
        onset_frames=2,
        offset_frames=2,
        min_frames=3,
        motion_on_threshold=0.01,
        motion_off_threshold=0.005,
    )

    def predict(buf):
        return [("test", 0.5)]

    seg = EnsembleSegmenter(cfg, predict, learned_model=None)
    t = 0.0
    events = []
    for i in range(20):
        m = 0.02 if 2 <= i < 12 else 0.0
        events.extend(seg.update(_wb_frame(i), m, t))
        t += 0.1
    assert any(e.kind == "start" for e in events)
