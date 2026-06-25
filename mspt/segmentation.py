"""Sign boundary segmentation for live MSPT — motion, confidence, learned, ensemble."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np
import torch

from mspt.normalize import flatten_xy
from mspt.rtmlib_io import streams_from_wholebody
from mspt.rtmlib_preprocess import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

PredictFn = Callable[[list[np.ndarray]], list[tuple[str, float]]]


class SegmentPhase(str, Enum):
    IDLE = "idle"
    SIGNING = "signing"
    COOLDOWN = "cooldown"
    # Fixed-mode legacy display phases
    RECORDING = "recording"
    HOLD = "hold"
    GAP = "gap"


@dataclass
class SegmentEvent:
    kind: Literal["start", "end"]
    timestamp: float
    frame_index: int
    buffer: list[np.ndarray] | None = None


@dataclass
class SegmenterConfig:
    min_frames: int = 8
    max_frames: int = 40
    max_clip_sec: float = 3.0
    cooldown_sec: float = 1.0
    hands_visible_min: float = 0.15
    motion_on_threshold: float = 0.012
    motion_off_threshold: float = 0.006
    onset_frames: int = 4
    offset_frames: int = 8
    trim_pad_frames: int = 2
    # Fixed timer mode
    clip_sec: float = 2.5
    gap_sec: float = 1.0
    hold_pred_sec: float = 5.0
    min_motion_frames: int = 6
    motion_threshold: float = 0.008
    # Confidence sliding window
    confidence_window_sec: float = 1.5
    confidence_stride_sec: float = 0.3
    conf_on: float = 0.12
    conf_off: float = 0.06
    stable_windows: int = 2
    unstable_windows: int = 2
    extract_fps: float = 10.0
    # Learned
    segmenter_checkpoint: Path | None = None
    prob_on: float = 0.55
    prob_off: float = 0.35
    device: str = "cpu"
    # Ensemble
    require_confidence_validation: bool = True


class SignSegmenter(Protocol):
    def reset(self) -> None: ...
    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]: ...
    @property
    def phase(self) -> SegmentPhase: ...
    @property
    def display_phase(self) -> str: ...
    @property
    def n_frames(self) -> int: ...
    @property
    def motion_frames(self) -> int: ...


def frame_motion(hands: np.ndarray, body: np.ndarray, prev: np.ndarray | None) -> float:
    cur = np.concatenate([hands.reshape(-1), body.reshape(-1)])
    if prev is None or prev.shape != cur.shape:
        return 0.0
    valid = (cur != 0) & (prev != 0)
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(cur[valid] - prev[valid])))


def hands_visible(wholebody: np.ndarray, min_score: float = 0.05) -> bool:
    lh = wholebody[LEFT_HAND_SLICE]
    rh = wholebody[RIGHT_HAND_SLICE]
    hands = np.concatenate([lh, rh], axis=0)
    if hands.size == 0:
        return False
    scores = hands[..., 2] if hands.shape[-1] >= 3 else hands[..., 3]
    valid = scores > min_score
    if not valid.any():
        return (hands[..., :2] != 0).any()
    return float(valid.mean()) >= 0.15


def trim_buffer_by_motion(
    buf: list[np.ndarray],
    motion_per_frame: list[float],
    motion_threshold: float,
    pad_frames: int = 2,
) -> list[np.ndarray]:
    if not buf:
        return []
    active = [m >= motion_threshold for m in motion_per_frame]
    if not any(active):
        return list(buf)
    start = next(i for i, a in enumerate(active) if a)
    end = len(active) - 1 - next(i for i, a in enumerate(reversed(active)) if a)
    start = max(0, start - pad_frames)
    end = min(len(buf) - 1, end + pad_frames)
    return buf[start : end + 1]


def resample_wholebody_buffer(buf: list[np.ndarray], max_seq_len: int) -> list[np.ndarray]:
    if not buf:
        return []
    seq = np.stack(buf, axis=0).astype(np.float32)
    hands, body, face, t = streams_from_wholebody(seq, max_seq_len)
    idx = np.linspace(0, len(buf) - 1, t, dtype=int)
    return [buf[i] for i in idx]


def wholebody_frame_features(wb: np.ndarray) -> np.ndarray:
    """Single-frame MSPT-normalized feature vector (294-d)."""
    seq = wb.reshape(1, *wb.shape).astype(np.float32)
    hands, body, face, _ = streams_from_wholebody(seq, max_frames=1)
    return np.concatenate(
        [flatten_xy(hands)[0], flatten_xy(body)[0], flatten_xy(face)[0]],
        axis=0,
    ).astype(np.float32)


def _finalize_segment(
    buf: list[np.ndarray],
    motions: list[float],
    cfg: SegmenterConfig,
    timestamp: float,
    frame_index: int,
) -> list[SegmentEvent]:
    trimmed = trim_buffer_by_motion(
        buf, motions, cfg.motion_off_threshold, pad_frames=cfg.trim_pad_frames,
    )
    if len(trimmed) < cfg.min_frames:
        return []
    return [
        SegmentEvent(
            kind="end",
            timestamp=timestamp,
            frame_index=frame_index,
            buffer=list(trimmed),
        )
    ]


class MotionSegmenter:
    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self._phase = SegmentPhase.IDLE
        self._buf: list[np.ndarray] = []
        self._motions: list[float] = []
        self._onset_count = 0
        self._offset_count = 0
        self._frame_index = 0
        self._sign_start_t = 0.0
        self._cooldown_start = 0.0
        self._motion_frames = 0

    def reset(self) -> None:
        self._phase = SegmentPhase.IDLE
        self._buf.clear()
        self._motions.clear()
        self._onset_count = 0
        self._offset_count = 0
        self._frame_index = 0
        self._motion_frames = 0

    @property
    def phase(self) -> SegmentPhase:
        return self._phase

    @property
    def display_phase(self) -> str:
        return self._phase.value

    @property
    def n_frames(self) -> int:
        return len(self._buf)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]:
        self._frame_index += 1
        events: list[SegmentEvent] = []

        if self._phase == SegmentPhase.COOLDOWN:
            if t - self._cooldown_start >= self.cfg.cooldown_sec:
                self._phase = SegmentPhase.IDLE
                self._onset_count = 0
                self._offset_count = 0
            return events

        visible = hands_visible(wb)
        moving = motion >= self.cfg.motion_on_threshold and visible

        if self._phase == SegmentPhase.IDLE:
            if moving:
                self._onset_count += 1
            else:
                self._onset_count = 0
            if self._onset_count >= self.cfg.onset_frames:
                self._phase = SegmentPhase.SIGNING
                self._buf = [wb.copy()]
                self._motions = [motion]
                self._motion_frames = 1
                self._sign_start_t = t
                self._offset_count = 0
                events.append(SegmentEvent("start", t, self._frame_index))
            return events

        # signing
        self._buf.append(wb.copy())
        self._motions.append(motion)
        if motion >= self.cfg.motion_off_threshold:
            self._motion_frames += 1
            self._offset_count = 0
        else:
            self._offset_count += 1

        elapsed = t - self._sign_start_t
        should_end = (
            self._offset_count >= self.cfg.offset_frames
            or not visible
            or elapsed >= self.cfg.max_clip_sec
            or len(self._buf) >= self.cfg.max_frames
        )
        if should_end:
            events.extend(
                _finalize_segment(self._buf, self._motions, self.cfg, t, self._frame_index)
            )
            self._buf.clear()
            self._motions.clear()
            self._motion_frames = 0
            self._phase = SegmentPhase.COOLDOWN
            self._cooldown_start = t
            self._onset_count = 0
            self._offset_count = 0
        return events


class ConfidenceSegmenter:
    def __init__(self, cfg: SegmenterConfig, predict_fn: PredictFn):
        self.cfg = cfg
        self._predict = predict_fn
        self._phase = SegmentPhase.IDLE
        self._ring: deque[np.ndarray] = deque(maxlen=max(40, int(cfg.extract_fps * 4)))
        self._motions: deque[float] = deque(maxlen=max(40, int(cfg.extract_fps * 4)))
        self._frame_index = 0
        self._frames_since_infer = 0
        self._window_frames = max(4, int(cfg.confidence_window_sec * cfg.extract_fps))
        self._stride_frames = max(1, int(cfg.confidence_stride_sec * cfg.extract_fps))
        self._history: list[tuple[str, float]] = []
        self._stable_count = 0
        self._unstable_count = 0
        self._segment_start_idx = 0
        self._cooldown_start = 0.0
        self._motion_frames = 0
        self._peak_conf = 0.0
        self._peak_label = ""

    def reset(self) -> None:
        self._phase = SegmentPhase.IDLE
        self._ring.clear()
        self._motions.clear()
        self._frame_index = 0
        self._frames_since_infer = 0
        self._history.clear()
        self._stable_count = 0
        self._unstable_count = 0
        self._motion_frames = 0

    @property
    def phase(self) -> SegmentPhase:
        return self._phase

    @property
    def display_phase(self) -> str:
        return self._phase.value

    @property
    def n_frames(self) -> int:
        return len(self._ring)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def _run_window(self) -> tuple[str, float] | None:
        if len(self._ring) < self._window_frames:
            return None
        window = list(self._ring)[-self._window_frames :]
        preds = self._predict(window)
        if not preds or preds[0][0] == "uncertain":
            return None
        return preds[0]

    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]:
        self._frame_index += 1
        events: list[SegmentEvent] = []
        self._ring.append(wb.copy())
        self._motions.append(motion)

        if self._phase == SegmentPhase.COOLDOWN:
            if t - self._cooldown_start >= self.cfg.cooldown_sec:
                self._phase = SegmentPhase.IDLE
            return events

        run_infer = (
            self._phase != SegmentPhase.IDLE or motion >= self.cfg.motion_on_threshold
        )
        if run_infer:
            self._frames_since_infer += 1
        if motion >= self.cfg.motion_off_threshold:
            self._motion_frames += 1

        pred: tuple[str, float] | None = None
        if run_infer and self._frames_since_infer >= self._stride_frames:
            self._frames_since_infer = 0
            pred = self._run_window()

        if self._phase == SegmentPhase.IDLE:
            if pred and pred[1] >= self.cfg.conf_on:
                if self._history and self._history[-1][0] == pred[0]:
                    self._stable_count += 1
                else:
                    self._stable_count = 1
                self._history.append(pred)
                if self._stable_count >= self.cfg.stable_windows:
                    self._phase = SegmentPhase.SIGNING
                    self._segment_start_idx = max(0, len(self._ring) - self._window_frames)
                    self._peak_conf = pred[1]
                    self._peak_label = pred[0]
                    events.append(SegmentEvent("start", t, self._frame_index))
            else:
                self._stable_count = 0
            return events

        # signing — check end conditions
        if pred:
            if pred[1] >= self._peak_conf:
                self._peak_conf = pred[1]
                self._peak_label = pred[0]
            if pred[1] < self.cfg.conf_off or pred[0] != self._peak_label:
                self._unstable_count += 1
            else:
                self._unstable_count = 0

        motion_still = motion < self.cfg.motion_off_threshold
        if (
            self._unstable_count >= self.cfg.unstable_windows
            or motion_still
            or len(self._ring) >= self.cfg.max_frames
        ):
            buf = list(self._ring)[self._segment_start_idx :]
            motions = list(self._motions)[self._segment_start_idx :]
            events.extend(_finalize_segment(buf, motions, self.cfg, t, self._frame_index))
            self._phase = SegmentPhase.COOLDOWN
            self._cooldown_start = t
            self._stable_count = 0
            self._unstable_count = 0
            self._history.clear()
            self._ring.clear()
            self._motions.clear()
            self._motion_frames = 0
        return events


class LearnedSegmenter:
    def __init__(self, cfg: SegmenterConfig, model: torch.nn.Module | None = None):
        self.cfg = cfg
        self._model = model
        self._phase = SegmentPhase.IDLE
        self._buf: list[np.ndarray] = []
        self._motions: list[float] = []
        self._feat_buf: list[np.ndarray] = []
        self._onset_count = 0
        self._offset_count = 0
        self._frame_index = 0
        self._sign_start_t = 0.0
        self._cooldown_start = 0.0
        self._motion_frames = 0
        self._device = torch.device(cfg.device)

    def reset(self) -> None:
        self._phase = SegmentPhase.IDLE
        self._buf.clear()
        self._motions.clear()
        self._feat_buf.clear()
        self._onset_count = 0
        self._offset_count = 0
        self._motion_frames = 0

    @property
    def phase(self) -> SegmentPhase:
        return self._phase

    @property
    def display_phase(self) -> str:
        return self._phase.value

    @property
    def n_frames(self) -> int:
        return len(self._buf)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def _sign_prob(self) -> float:
        if self._model is None or not self._feat_buf:
            return 0.0
        x = torch.from_numpy(np.stack(self._feat_buf, axis=0)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            probs = self._model.predict_proba(x)
        return float(probs[-1].item())

    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]:
        self._frame_index += 1
        events: list[SegmentEvent] = []
        feat = wholebody_frame_features(wb)
        self._feat_buf.append(feat)
        if len(self._feat_buf) > 64:
            self._feat_buf.pop(0)

        if self._phase == SegmentPhase.COOLDOWN:
            if t - self._cooldown_start >= self.cfg.cooldown_sec:
                self._phase = SegmentPhase.IDLE
            return events

        prob = self._sign_prob()

        if self._phase == SegmentPhase.IDLE:
            if prob >= self.cfg.prob_on:
                self._onset_count += 1
            else:
                self._onset_count = 0
            if self._onset_count >= self.cfg.onset_frames:
                self._phase = SegmentPhase.SIGNING
                self._buf = [wb.copy()]
                self._motions = [motion]
                self._motion_frames = 1
                self._sign_start_t = t
                self._offset_count = 0
                events.append(SegmentEvent("start", t, self._frame_index))
            return events

        self._buf.append(wb.copy())
        self._motions.append(motion)
        if motion >= self.cfg.motion_off_threshold:
            self._motion_frames += 1

        if prob < self.cfg.prob_off:
            self._offset_count += 1
        else:
            self._offset_count = 0

        elapsed = t - self._sign_start_t
        if (
            self._offset_count >= self.cfg.offset_frames
            or elapsed >= self.cfg.max_clip_sec
            or len(self._buf) >= self.cfg.max_frames
        ):
            events.extend(
                _finalize_segment(self._buf, self._motions, self.cfg, t, self._frame_index)
            )
            self._buf.clear()
            self._motions.clear()
            self._motion_frames = 0
            self._phase = SegmentPhase.COOLDOWN
            self._cooldown_start = t
            self._onset_count = 0
            self._offset_count = 0
        return events


class EnsembleSegmenter:
    """Motion/learned start, learned+motion end, optional confidence validation."""

    def __init__(
        self,
        cfg: SegmenterConfig,
        predict_fn: PredictFn,
        learned_model: torch.nn.Module | None = None,
    ):
        self.cfg = cfg
        self._predict = predict_fn
        self._learned_model = learned_model
        self._device = torch.device(cfg.device)
        self._phase = SegmentPhase.IDLE
        self._buf: list[np.ndarray] = []
        self._motions: list[float] = []
        self._feat_buf: list[np.ndarray] = []
        self._frame_index = 0
        self._cooldown_start = 0.0
        self._motion_frames = 0
        self._motion_onset = 0
        self._learned_onset = 0
        self._motion_offset = 0
        self._learned_offset = 0
        self._sign_start_t = 0.0

    def reset(self) -> None:
        self._phase = SegmentPhase.IDLE
        self._buf.clear()
        self._motions.clear()
        self._feat_buf.clear()
        self._motion_frames = 0
        self._motion_onset = 0
        self._learned_onset = 0
        self._motion_offset = 0
        self._learned_offset = 0

    @property
    def phase(self) -> SegmentPhase:
        return self._phase

    @property
    def display_phase(self) -> str:
        return self._phase.value

    @property
    def n_frames(self) -> int:
        return len(self._buf)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def _learned_prob(self, wb: np.ndarray) -> float:
        if self._learned_model is None:
            return 0.0
        feat = wholebody_frame_features(wb)
        self._feat_buf.append(feat)
        if len(self._feat_buf) > 64:
            self._feat_buf.pop(0)
        x = torch.from_numpy(np.stack(self._feat_buf, axis=0)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            probs = self._learned_model.predict_proba(x)
        return float(probs[-1].item())

    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]:
        self._frame_index += 1
        events: list[SegmentEvent] = []

        if self._phase == SegmentPhase.COOLDOWN:
            if t - self._cooldown_start >= self.cfg.cooldown_sec:
                self._phase = SegmentPhase.IDLE
            return events

        visible = hands_visible(wb)
        moving = motion >= self.cfg.motion_on_threshold and visible
        prob = self._learned_prob(wb) if self._learned_model is not None else 0.0

        if self._phase == SegmentPhase.IDLE:
            if moving:
                self._motion_onset += 1
            else:
                self._motion_onset = 0
            if prob >= self.cfg.prob_on:
                self._learned_onset += 1
            else:
                self._learned_onset = 0
            start_motion = self._motion_onset >= self.cfg.onset_frames
            start_learned = self._learned_onset >= self.cfg.onset_frames
            if start_motion or start_learned:
                self._phase = SegmentPhase.SIGNING
                self._buf = [wb.copy()]
                self._motions = [motion]
                self._motion_frames = 1 if moving else 0
                self._sign_start_t = t
                self._motion_offset = 0
                self._learned_offset = 0
                events.append(SegmentEvent("start", t, self._frame_index))
            return events

        self._buf.append(wb.copy())
        self._motions.append(motion)
        if moving:
            self._motion_frames += 1
            self._motion_offset = 0
        else:
            self._motion_offset += 1
        if prob < self.cfg.prob_off:
            self._learned_offset += 1
        else:
            self._learned_offset = 0

        motion_end = self._motion_offset >= self.cfg.offset_frames
        learned_end = (
            self._learned_model is not None
            and self._learned_offset >= self.cfg.offset_frames
        )
        elapsed = t - self._sign_start_t
        if self._learned_model is None:
            should_end = motion_end or elapsed >= self.cfg.max_clip_sec or len(self._buf) >= self.cfg.max_frames
        else:
            should_end = (
                (learned_end and motion_end)
                or elapsed >= self.cfg.max_clip_sec
                or len(self._buf) >= self.cfg.max_frames
            )
        if should_end:
            buf = list(self._buf)
            motions = list(self._motions)
            if self.cfg.require_confidence_validation and len(buf) >= self.cfg.min_frames:
                preds = self._predict(buf)
                if not preds or preds[0][0] == "uncertain":
                    buf = []
            if buf:
                events.extend(_finalize_segment(buf, motions, self.cfg, t, self._frame_index))
            self._buf.clear()
            self._motions.clear()
            self._feat_buf.clear()
            self._motion_frames = 0
            self._phase = SegmentPhase.COOLDOWN
            self._cooldown_start = t
            self._motion_onset = 0
            self._learned_onset = 0
        return events


class FixedSegmenter:
    """Legacy timer-based recording / hold / gap (unchanged default behavior)."""

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self._phase = SegmentPhase.RECORDING
        self._buf: list[np.ndarray] = []
        self._motions: list[float] = []
        self._frame_index = 0
        self._clip_start = 0.0
        self._gap_start = 0.0
        self._hold_start = 0.0
        self._motion_frames = 0
        self._pending_end = False
        self._initialized = False

    def reset(self) -> None:
        self._phase = SegmentPhase.RECORDING
        self._buf.clear()
        self._motions.clear()
        self._motion_frames = 0
        self._pending_end = False

    def start_session(self, t: float) -> None:
        self._clip_start = t
        self._initialized = True

    @property
    def phase(self) -> SegmentPhase:
        return self._phase

    @property
    def display_phase(self) -> str:
        return self._phase.value

    @property
    def n_frames(self) -> int:
        return len(self._buf)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def enter_hold(self, t: float) -> None:
        self._phase = SegmentPhase.HOLD
        self._hold_start = t

    def enter_gap(self, t: float) -> None:
        self._phase = SegmentPhase.GAP
        self._gap_start = t
        self._buf.clear()
        self._motions.clear()
        self._motion_frames = 0

    def enter_recording(self, t: float) -> None:
        self._phase = SegmentPhase.RECORDING
        self._clip_start = t

    def update(self, wb: np.ndarray, motion: float, t: float) -> list[SegmentEvent]:
        self._frame_index += 1
        events: list[SegmentEvent] = []
        moving = motion >= self.cfg.motion_threshold

        if self._phase == SegmentPhase.RECORDING:
            if moving:
                self._buf.append(wb.copy())
                self._motions.append(motion)
                self._motion_frames += 1
            elapsed = t - self._clip_start
            if elapsed >= self.cfg.clip_sec:
                if (
                    self._motion_frames >= self.cfg.min_motion_frames
                    and len(self._buf) >= self.cfg.min_frames
                ):
                    events.append(
                        SegmentEvent(
                            kind="end",
                            timestamp=t,
                            frame_index=self._frame_index,
                            buffer=list(self._buf),
                        )
                    )
                self._buf.clear()
                self._motions.clear()
                self._motion_frames = 0
                self.enter_hold(t)
        elif self._phase == SegmentPhase.HOLD:
            if t - self._hold_start >= self.cfg.hold_pred_sec:
                self.enter_gap(t)
        elif self._phase == SegmentPhase.GAP:
            if t - self._gap_start >= self.cfg.gap_sec:
                self.enter_recording(t)
        return events


def load_learned_model(checkpoint: Path, device: str) -> torch.nn.Module:
    from mspt.segmenter_model import SignFrameClassifier

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim", 294))
    model = SignFrameClassifier(input_dim=input_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def segmenter_config_from_args(args) -> SegmenterConfig:
    cooldown = getattr(args, "cooldown_sec", None)
    if cooldown is None:
        cooldown = getattr(args, "gap_sec", 1.0)
    return SegmenterConfig(
        min_frames=getattr(args, "min_frames", 8),
        max_frames=getattr(args, "max_frames", 40),
        max_clip_sec=getattr(args, "max_clip_sec", 3.0),
        cooldown_sec=cooldown,
        motion_on_threshold=getattr(args, "motion_on_threshold", 0.012),
        motion_off_threshold=getattr(args, "motion_off_threshold", 0.006),
        onset_frames=getattr(args, "onset_frames", 4),
        offset_frames=getattr(args, "offset_frames", 8),
        clip_sec=getattr(args, "clip_sec", 2.5),
        gap_sec=getattr(args, "gap_sec", 1.0),
        hold_pred_sec=getattr(args, "hold_pred_sec", 5.0),
        min_motion_frames=getattr(args, "min_motion_frames", 6),
        motion_threshold=getattr(args, "motion_threshold", 0.008),
        confidence_window_sec=getattr(args, "confidence_window_sec", 1.5),
        confidence_stride_sec=getattr(args, "confidence_stride_sec", 0.3),
        conf_on=getattr(args, "conf_on", getattr(args, "min_confidence", 0.12)),
        conf_off=getattr(args, "conf_off", 0.06),
        stable_windows=getattr(args, "stable_windows", 2),
        unstable_windows=getattr(args, "unstable_windows", 2),
        extract_fps=float(getattr(args, "fps", 10)),
        segmenter_checkpoint=getattr(args, "segmenter_checkpoint", None),
        prob_on=getattr(args, "prob_on", 0.55),
        prob_off=getattr(args, "prob_off", 0.35),
        device=getattr(args, "device", "cpu"),
    )


def add_segmenter_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--segmenter",
        choices=("fixed", "motion", "confidence", "learned", "ensemble"),
        default="fixed",
        help="Sign boundary strategy (default: fixed 2.5s timer)",
    )
    ap.add_argument("--motion-on-threshold", type=float, default=0.012)
    ap.add_argument("--motion-off-threshold", type=float, default=0.006)
    ap.add_argument("--onset-frames", type=int, default=4)
    ap.add_argument("--offset-frames", type=int, default=8)
    ap.add_argument("--max-clip-sec", type=float, default=3.0)
    ap.add_argument("--cooldown-sec", type=float, default=None,
                    help="Post-segment pause (defaults to --gap-sec)")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--confidence-window-sec", type=float, default=1.5)
    ap.add_argument("--confidence-stride-sec", type=float, default=0.3)
    ap.add_argument("--conf-on", type=float, default=None, help="Confidence segmenter onset")
    ap.add_argument("--conf-off", type=float, default=0.06)
    ap.add_argument("--stable-windows", type=int, default=2)
    ap.add_argument("--unstable-windows", type=int, default=2)
    ap.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=None,
        help="Learned segmenter weights (checkpoints/mspt/sign_segmenter_best.pt)",
    )
    ap.add_argument("--prob-on", type=float, default=0.55)
    ap.add_argument("--prob-off", type=float, default=0.35)


def make_segmenter(
    mode: str,
    cfg: SegmenterConfig,
    predict_fn: PredictFn | None = None,
) -> SignSegmenter:
    if mode == "fixed":
        return FixedSegmenter(cfg)
    if mode == "motion":
        return MotionSegmenter(cfg)
    if mode == "confidence":
        if predict_fn is None:
            raise ValueError("confidence segmenter requires predict_fn")
        return ConfidenceSegmenter(cfg, predict_fn)
    if mode == "learned":
        model = None
        if cfg.segmenter_checkpoint and Path(cfg.segmenter_checkpoint).is_file():
            model = load_learned_model(Path(cfg.segmenter_checkpoint), cfg.device)
        elif cfg.segmenter_checkpoint:
            print(f"[segmenter] checkpoint not found: {cfg.segmenter_checkpoint} — motion fallback")
        return LearnedSegmenter(cfg, model)
    if mode == "ensemble":
        learned = None
        if cfg.segmenter_checkpoint and Path(cfg.segmenter_checkpoint).is_file():
            learned = load_learned_model(Path(cfg.segmenter_checkpoint), cfg.device)
        elif cfg.segmenter_checkpoint:
            print(f"[segmenter] ensemble: no checkpoint, motion+confidence only")
        if predict_fn is None:
            raise ValueError("ensemble segmenter requires predict_fn")
        return EnsembleSegmenter(cfg, predict_fn, learned)
    raise ValueError(f"unknown segmenter mode: {mode}")
