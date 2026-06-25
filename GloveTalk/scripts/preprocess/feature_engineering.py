"""Engineer 40-dim feature vectors from 18-dim raw glove sensor frames."""
import json
from pathlib import Path

import numpy as np

RAW_DIM = 18
REL_DIM = 4
DERIV_DIM = 18
NUM_FEATURES = 40

L_QUAT = slice(0, 4)
R_QUAT = slice(9, 13)
FLEX_INDICES = [4, 5, 6, 7, 8, 13, 14, 15, 16, 17]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def relative_orientation(raw_frame: np.ndarray) -> np.ndarray:
    q_l = normalize_quaternion(raw_frame[L_QUAT])
    q_r = normalize_quaternion(raw_frame[R_QUAT])
    q_l_conj = np.array([q_l[0], -q_l[1], -q_l[2], -q_l[3]], dtype=np.float32)
    return normalize_quaternion(quaternion_multiply(q_r, q_l_conj))


def preprocess_raw_frame(raw_frame: np.ndarray) -> np.ndarray:
    frame = np.array(raw_frame, dtype=np.float32).copy()
    frame[L_QUAT] = normalize_quaternion(frame[L_QUAT])
    frame[R_QUAT] = normalize_quaternion(frame[R_QUAT])
    for idx in FLEX_INDICES:
        frame[idx] = np.clip(frame[idx], 0.0, 1.0)
    return frame


def compute_derivatives(raw_sequence: np.ndarray, dt: float = 0.02) -> np.ndarray:
    diffs = np.zeros_like(raw_sequence, dtype=np.float32)
    if len(raw_sequence) > 1:
        diffs[1:] = (raw_sequence[1:] - raw_sequence[:-1]) / dt
    return diffs


def raw_sequence_to_features(raw_sequence: np.ndarray, dt: float = 0.02) -> np.ndarray:
    if raw_sequence.ndim == 1:
        raw_sequence = raw_sequence.reshape(1, -1)

    cleaned = np.stack([preprocess_raw_frame(f) for f in raw_sequence])
    rel_orient = np.stack([relative_orientation(f) for f in cleaned])
    derivatives = compute_derivatives(cleaned, dt=dt)
    return np.concatenate([cleaned, rel_orient, derivatives], axis=1).astype(np.float32)


def sliding_windows(feature_sequence: np.ndarray, window_size: int, stride: int) -> list:
    windows = []
    total = len(feature_sequence)
    for start in range(0, total - window_size + 1, stride):
        windows.append(feature_sequence[start : start + window_size])
    return windows


def save_feature_config(path: Path, window_size: int, dt: float) -> None:
    config = {
        "num_raw_features": RAW_DIM,
        "num_relative_features": REL_DIM,
        "num_derivative_features": DERIV_DIM,
        "num_features": NUM_FEATURES,
        "window_size": window_size,
        "dt": dt,
        "feature_layout": "raw_18 + relative_quat_4 + derivatives_18",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
