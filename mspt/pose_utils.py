"""MediaPipe pose/hand landmark helpers (no include50_lab import required)."""

from __future__ import annotations

import numpy as np
from pathlib import Path

# Match include50_lab/preprocess/skeleton.py + config.py
SCALE_INVARIANCE = True
SCALE_TARGET_SHOULDER = 0.25
UPPER_POSE_LANDMARKS = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

POSE_IDXS = sorted(UPPER_POSE_LANDMARKS)
NUM_POSE = len(POSE_IDXS)
NUM_HAND = 21
NUM_LANDMARKS = NUM_POSE + 2 * NUM_HAND
NUM_BODY = 33


def _get_scale_transform(pose_result):
    if not SCALE_INVARIANCE or not pose_result or not pose_result.pose_landmarks:
        return None
    landmarks = pose_result.pose_landmarks[0]
    if len(landmarks) <= 12:
        return None
    ls, rs = landmarks[11], landmarks[12]
    if not all(0 <= v <= 1 for v in (ls.x, ls.y, rs.x, rs.y)):
        return None
    dist = ((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5
    if dist <= 1e-6:
        return None
    cx, cy = (ls.x + rs.x) * 0.5, (ls.y + rs.y) * 0.5
    return cx, cy, SCALE_TARGET_SHOULDER / dist


def _apply_scale(x: float, y: float, transform) -> tuple[float, float]:
    if transform is None:
        return x, y
    cx, cy, scale = transform
    nx = max(0.0, min(1.0, (x - cx) * scale + cx))
    ny = max(0.0, min(1.0, (y - cy) * scale + cy))
    return nx, ny


def landmarks_from_results(pose_result, hand_result) -> np.ndarray:
    """Return ``(54, 4)`` — 12 upper pose + 21 left hand + 21 right hand."""
    out = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    scale_tf = _get_scale_transform(pose_result)

    if pose_result and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        for j, idx in enumerate(POSE_IDXS):
            if idx < len(lms):
                lm = lms[idx]
                x, y = _apply_scale(lm.x, lm.y, scale_tf)
                out[j] = (x, y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)

    left_offset = NUM_POSE
    right_offset = NUM_POSE + NUM_HAND
    if hand_result and hand_result.hand_landmarks:
        for h_idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness = "Right"
            if hand_result.handedness and h_idx < len(hand_result.handedness):
                handedness = hand_result.handedness[h_idx][0].category_name
            base = left_offset if handedness == "Left" else right_offset
            for k, lm in enumerate(landmarks):
                if k >= NUM_HAND:
                    break
                x, y = _apply_scale(lm.x, lm.y, scale_tf)
                out[base + k] = (x, y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
    return out


def body_from_pose(pose_result) -> np.ndarray:
    """Full 33-pose joints ``(33, 4)`` with shoulder scale invariance."""
    out = np.zeros((NUM_BODY, 4), dtype=np.float32)
    scale_tf = _get_scale_transform(pose_result)
    if pose_result and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        for j in range(min(NUM_BODY, len(lms))):
            lm = lms[j]
            x, y = _apply_scale(lm.x, lm.y, scale_tf)
            out[j] = (x, y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
    return out


def body_ready(lab_root: Path, min_fraction: float = 0.9) -> bool:
    lm = lab_root / "cache" / "landmarks"
    body = lab_root / "cache" / "mspt_body"
    n_lm = sum(1 for _ in lm.rglob("*.npy")) if lm.is_dir() else 0
    n_body = sum(1 for _ in body.rglob("*.npy")) if body.is_dir() else 0
    if n_lm == 0:
        return n_body > 0
    return n_body >= min_fraction * n_lm
