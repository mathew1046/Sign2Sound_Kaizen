"""Per-stream 2D keypoint normalization for MSPT."""

from __future__ import annotations

import numpy as np

# MediaPipe pose indices
POSE_LHIP, POSE_RHIP = 23, 24
POSE_LSHOULDER, POSE_RSHOULDER = 11, 12
POSE_NOSE = 0

# Hand indices (per hand, 21 joints)
HAND_WRIST = 0
HAND_MIDDLE_TIP = 12

# Face mesh indices (full 468 mesh) for anchor / scale
FACE_NOSE_TIP = 1
FACE_L_EYE = 33
FACE_R_EYE = 263


def _anchor_scale(xy: np.ndarray, anchor_idx: int, scale_idxs: tuple[int, int], eps: float = 1e-6) -> np.ndarray:
    """Centre on ``anchor_idx`` and scale by distance between ``scale_idxs``."""
    out = xy.copy()
    T, V, _ = out.shape
    for t in range(T):
        frame = out[t]
        valid = (frame != 0).any(axis=-1)
        if not valid.any():
            continue
        anchor = frame[anchor_idx]
        if (anchor == 0).all():
            # fallback: centroid of valid joints
            anchor = frame[valid].mean(axis=0)
        p0, p1 = frame[scale_idxs[0]], frame[scale_idxs[1]]
        span = float(np.linalg.norm(p1 - p0))
        if span < eps:
            # fallback span from valid extent
            pts = frame[valid]
            span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), eps)
        out[t] = (frame - anchor) / span
    return out


def normalize_hands(hands: np.ndarray) -> np.ndarray:
    """``(T, 42, 2)`` left+right hand joints, wrist-anchored, hand-span scaled."""
    T = hands.shape[0]
    lh = hands[:, :21]
    rh = hands[:, 21:]
    lh = _anchor_scale(lh, HAND_WRIST, (HAND_WRIST, HAND_MIDDLE_TIP))
    rh = _anchor_scale(rh, HAND_WRIST, (HAND_WRIST, HAND_MIDDLE_TIP))
    return np.concatenate([lh, rh], axis=1)


def normalize_body(body: np.ndarray) -> np.ndarray:
    """``(T, 33, 2)`` pose joints, hip-midpoint anchored, shoulder-width scaled."""
    out = body.copy()
    T = out.shape[0]
    for t in range(T):
        frame = out[t]
        valid = (frame != 0).any(axis=-1)
        if not valid.any():
            continue
        lh, rh = frame[POSE_LHIP], frame[POSE_RHIP]
        if (lh == 0).all() or (rh == 0).all():
            anchor = frame[valid].mean(axis=0)
        else:
            anchor = (lh + rh) * 0.5
        ls, rs = frame[POSE_LSHOULDER], frame[POSE_RSHOULDER]
        span = float(np.linalg.norm(rs - ls))
        if span < 1e-6:
            span = max(float(np.ptp(frame[valid, 0])), float(np.ptp(frame[valid, 1])), 1e-6)
        out[t] = (frame - anchor) / span
    return out


def normalize_face_from_mesh(face: np.ndarray, face_mesh_idxs: tuple[int, ...] | None = None) -> np.ndarray:
    """``(T, V, 2)`` face subset; nose anchored, inter-eye scaled.

    When ``face_mesh_idxs`` maps subset positions to full mesh ids, use those for
    anchor/scale. Otherwise assume standard 468-mesh indices are stored in order.
    """
    out = face.copy()
    idx_map = {i: (face_mesh_idxs[i] if face_mesh_idxs else i) for i in range(face.shape[1])}

    def _mesh_id(subset_i: int) -> int:
        return idx_map.get(subset_i, subset_i)

    # find subset positions matching nose / eyes
    nose_pos = next((i for i, mid in idx_map.items() if mid == FACE_NOSE_TIP), 0)
    leye_pos = next((i for i, mid in idx_map.items() if mid == FACE_L_EYE), None)
    reye_pos = next((i for i, mid in idx_map.items() if mid == FACE_R_EYE), None)

    T = out.shape[0]
    for t in range(T):
        frame = out[t]
        valid = (frame != 0).any(axis=-1)
        if not valid.any():
            continue
        anchor = frame[nose_pos]
        if (anchor == 0).all():
            anchor = frame[valid].mean(axis=0)
        if leye_pos is not None and reye_pos is not None:
            span = float(np.linalg.norm(frame[reye_pos] - frame[leye_pos]))
        else:
            span = max(float(np.ptp(frame[valid, 0])), float(np.ptp(frame[valid, 1])), 1e-6)
        if span < 1e-6:
            span = 1e-6
        out[t] = (frame - anchor) / span
    return out


def flatten_xy(seq: np.ndarray) -> np.ndarray:
    """``(T, V, 2)`` -> ``(T, V*2)``."""
    if seq.size == 0:
        return np.zeros((1, seq.shape[1] * 2 if seq.ndim >= 2 else 0), dtype=np.float32)
    return seq.reshape(seq.shape[0], -1).astype(np.float32)
