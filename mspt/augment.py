"""Temporal keypoint augmentation for MSPT (target ~8-10× effective expansion)."""

from __future__ import annotations

import numpy as np


def gaussian_jitter(seq: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    out = seq.copy()
    mask = out != 0
    noise = np.random.randn(*out.shape).astype(np.float32) * sigma
    out = out + noise * mask
    return out


def skeleton_scale(seq: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    s = np.random.uniform(low, high)
    out = seq.copy()
    valid = out != 0
    out[valid] *= s
    return out


def temporal_dropout(seq: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
    out = seq.copy()
    T = out.shape[0]
    for t in range(T):
        if np.random.rand() < drop_prob:
            out[t] = 0.0
    return out


def temporal_resample(seq: np.ndarray, low: float = 0.8, high: float = 1.2, max_len: int | None = None) -> np.ndarray:
    T = seq.shape[0]
    if T < 2:
        return seq.copy()
    new_T = max(2, int(T * np.random.uniform(low, high)))
    if max_len is not None:
        new_T = min(new_T, max_len)
    src = np.linspace(0, T - 1, new_T)
    idx_lo = np.floor(src).astype(int)
    idx_hi = np.minimum(idx_lo + 1, T - 1)
    w = (src - idx_lo).astype(np.float32)
    if seq.ndim == 2:
        out = (1 - w)[:, None] * seq[idx_lo] + w[:, None] * seq[idx_hi]
    else:
        w = w[:, None, None]
        out = (1 - w) * seq[idx_lo] + w * seq[idx_hi]
    if max_len is not None and len(out) > max_len:
        out = out[:max_len]
    return out.astype(np.float32)


def horizontal_flip_hands(hands: np.ndarray) -> np.ndarray:
    """Flip x and swap left/right hand blocks ``(T, 42, 2)``."""
    out = hands.copy()
    out[..., 0] *= -1
    lh, rh = out[:, :21].copy(), out[:, 21:].copy()
    out[:, :21] = rh
    out[:, 21:] = lh
    return out


def horizontal_flip_body(body: np.ndarray) -> np.ndarray:
    out = body.copy()
    out[..., 0] *= -1
    # swap left/right symmetric pairs (MediaPipe pose)
    swap_pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
    ]
    for a, b in swap_pairs:
        if a < body.shape[1] and b < body.shape[1]:
            tmp = out[:, a].copy()
            out[:, a] = out[:, b]
            out[:, b] = tmp
    return out


def horizontal_flip_face(face: np.ndarray) -> np.ndarray:
    out = face.copy()
    out[..., 0] *= -1
    return out


def spoter_perspective(seq: np.ndarray, max_shift: float = 0.15, max_scale: float = 0.2) -> np.ndarray:
    """SPOTER-style weak perspective on 2D coords (SLR augmentation)."""
    out = seq.copy()
    T = out.shape[0]
    if T == 0:
        return out
    sx = 1.0 + np.random.uniform(-max_scale, max_scale)
    sy = 1.0 + np.random.uniform(-max_scale, max_scale)
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)
    for t in range(T):
        frame = out[t]
        if frame.ndim != 2:
            continue
        valid = (frame != 0).any(axis=-1)
        if not valid.any():
            continue
        frame[valid, 0] = frame[valid, 0] * sx + tx
        frame[valid, 1] = frame[valid, 1] * sy + ty
        out[t] = frame
    return out


def apply_augmentations(
    hand: np.ndarray,
    body: np.ndarray,
    face: np.ndarray,
    training: bool = True,
    max_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not training:
        return hand, body, face

    if np.random.rand() < 0.5:
        hand = horizontal_flip_hands(hand)
        body = horizontal_flip_body(body)
        face = horizontal_flip_face(face)

    hand = gaussian_jitter(hand)
    body = gaussian_jitter(body)
    face = gaussian_jitter(face)

    hand = skeleton_scale(hand)
    body = skeleton_scale(body)
    face = skeleton_scale(face)

    hand = temporal_dropout(hand)
    body = temporal_dropout(body)
    face = temporal_dropout(face)

    # resample with same factor for alignment
    factor = np.random.uniform(0.8, 1.2)
    T = hand.shape[0]
    new_T = max(2, int(T * factor))
    if max_len is not None:
        new_T = min(new_T, max_len)
    src = np.linspace(0, T - 1, new_T)
    idx_lo = np.floor(src).astype(int)
    idx_hi = np.minimum(idx_lo + 1, T - 1)
    w = (src - idx_lo).astype(np.float32)

    def _resample_aligned(seq):
        if seq.ndim == 2:
            return ((1 - w)[:, None] * seq[idx_lo] + w[:, None] * seq[idx_hi]).astype(np.float32)
        w3 = w[:, None, None]
        return ((1 - w3) * seq[idx_lo] + w3 * seq[idx_hi]).astype(np.float32)

    hand = _resample_aligned(hand)
    body = _resample_aligned(body)
    face = _resample_aligned(face)

    hand = spoter_perspective(hand)
    body = spoter_perspective(body)
    face = spoter_perspective(face)

    return hand, body, face
