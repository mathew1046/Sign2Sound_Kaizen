"""Load + normalize rtmlib COCO-WholeBody caches for MSPT (same tensor shapes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mspt.normalize import _anchor_scale, normalize_hands

NUM_HAND_KP = 42
NUM_BODY_KP = 33
NUM_FACE_KP = 72  # MSPT face stream width (pad COCO-68)

# COCO-17 body indices
COCO_LHIP, COCO_RHIP = 11, 12
COCO_LSHOULDER, COCO_RSHOULDER = 5, 6
# iBUG-68 face
FACE_NOSE, FACE_L_EYE, FACE_R_EYE = 30, 36, 45


def _xy_from_npy(arr: np.ndarray) -> np.ndarray:
    """(T, V, 4) normalized xy with invalid joints zeroed."""
    xy = np.array(arr[..., :2], dtype=np.float32, copy=True)
    if arr.shape[-1] >= 4:
        valid = arr[..., 3] > 0
        xy[~valid] = 0.0
    return xy


def _subsample_frames(arr: np.ndarray, max_frames: int) -> np.ndarray:
    t = len(arr)
    if t <= max_frames:
        return arr
    idx = np.linspace(0, t - 1, max_frames, dtype=int)
    return arr[idx]


def normalize_body_coco17(body: np.ndarray) -> np.ndarray:
    """``(T, 17, 2)`` COCO body -> hip anchor, shoulder span; then pad to 33."""
    out = body.copy()
    for t in range(out.shape[0]):
        frame = out[t]
        valid = (frame != 0).any(axis=-1)
        if not valid.any():
            continue
        lh, rh = frame[COCO_LHIP], frame[COCO_RHIP]
        if (lh == 0).all() or (rh == 0).all():
            anchor = frame[valid].mean(axis=0)
        else:
            anchor = (lh + rh) * 0.5
        ls, rs = frame[COCO_LSHOULDER], frame[COCO_RSHOULDER]
        span = float(np.linalg.norm(rs - ls))
        if span < 1e-6:
            span = max(float(np.ptp(frame[valid, 0])), float(np.ptp(frame[valid, 1])), 1e-6)
        out[t] = (frame - anchor) / span
    padded = np.zeros((out.shape[0], NUM_BODY_KP, 2), dtype=np.float32)
    padded[:, : out.shape[1]] = out
    return padded


def normalize_face_coco68(face: np.ndarray) -> np.ndarray:
    """``(T, 68, 2)`` -> normalize and pad to 72 for MSPT face_dim."""
    normed = _anchor_scale(face, FACE_NOSE, (FACE_L_EYE, FACE_R_EYE))
    padded = np.zeros((normed.shape[0], NUM_FACE_KP, 2), dtype=np.float32)
    padded[:, : normed.shape[1]] = normed
    return padded


def streams_from_wholebody(
    wholebody: np.ndarray,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(T, 133, 4)`` wholebody buffer -> MSPT-normalized hand/body/face streams."""
    from mspt.rtmlib_preprocess import (
        BODY_SLICE,
        FACE_SLICE,
        LEFT_HAND_SLICE,
        RIGHT_HAND_SLICE,
    )

    lh = _xy_from_npy(wholebody[:, LEFT_HAND_SLICE])
    rh = _xy_from_npy(wholebody[:, RIGHT_HAND_SLICE])
    body17 = _xy_from_npy(wholebody[:, BODY_SLICE])
    face68 = _xy_from_npy(wholebody[:, FACE_SLICE])
    hands = np.concatenate([lh, rh], axis=1)

    hands = _subsample_frames(hands, max_frames)
    body = _subsample_frames(body17, max_frames)
    face = _subsample_frames(face68, max_frames)

    hands = normalize_hands(hands)
    body = normalize_body_coco17(body)
    face = normalize_face_coco68(face)
    return hands, body, face, len(hands)


def load_streams_rtmlib(
    left_hand_path: Path,
    right_hand_path: Path,
    body_path: Path,
    face_path: Path,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    lh = _xy_from_npy(np.load(left_hand_path, mmap_mode="r"))
    rh = _xy_from_npy(np.load(right_hand_path, mmap_mode="r"))
    body17 = _xy_from_npy(np.load(body_path, mmap_mode="r"))
    face68 = _xy_from_npy(np.load(face_path, mmap_mode="r"))

    t_raw = min(len(lh), len(rh), len(body17), len(face68))
    lh, rh, body17, face68 = lh[:t_raw], rh[:t_raw], body17[:t_raw], face68[:t_raw]
    hands = np.concatenate([lh, rh], axis=1)

    hands = _subsample_frames(hands, max_frames)
    body = _subsample_frames(body17, max_frames)
    face = _subsample_frames(face68, max_frames)

    hands = normalize_hands(hands)
    body = normalize_body_coco17(body)
    face = normalize_face_coco68(face)

    t = len(hands)
    return hands, body, face, t
