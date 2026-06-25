"""RTMLIB wholebody landmarks -> orientation feature vectors."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mspt.rtmlib_preprocess import BODY_SLICE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE

CONF_THRESH = 0.5

# COCO-17 body joint indices (within body slice)
BODY_L_SHOULDER, BODY_R_SHOULDER = 5, 6
BODY_L_ELBOW, BODY_R_ELBOW = 7, 8
BODY_L_WRIST, BODY_R_WRIST = 9, 10

# Hand joint indices (per 21-joint hand)
HAND_WRIST = 0
HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP = 1, 2, 3
HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP = 5, 6, 7
HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP = 9, 10, 11
HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP = 13, 14, 15
HAND_PINKY_MCP, HAND_PINKY_PIP, HAND_PINKY_DIP = 17, 18, 19

FINGER_CURL_JOINTS: dict[str, list[tuple[int, int, int]]] = {
    "thumb": [(HAND_WRIST, HAND_THUMB_CMC, HAND_THUMB_MCP), (HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP)],
    "index": [
        (HAND_WRIST, HAND_INDEX_MCP, HAND_INDEX_PIP),
        (HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP),
        (HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_DIP + 1),
    ],
    "middle": [
        (HAND_WRIST, HAND_MIDDLE_MCP, HAND_MIDDLE_PIP),
        (HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP),
        (HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_DIP + 1),
    ],
    "ring": [
        (HAND_WRIST, HAND_RING_MCP, HAND_RING_PIP),
        (HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP),
        (HAND_RING_PIP, HAND_RING_DIP, HAND_RING_DIP + 1),
    ],
    "pinky": [
        (HAND_WRIST, HAND_PINKY_MCP, HAND_PINKY_PIP),
        (HAND_PINKY_MCP, HAND_PINKY_PIP, HAND_PINKY_DIP),
        (HAND_PINKY_PIP, HAND_PINKY_DIP, HAND_PINKY_DIP + 1),
    ],
}


def _valid_joint(frame: np.ndarray, idx: int) -> bool:
    if idx < 0 or idx >= len(frame):
        return False
    if frame.shape[-1] >= 4:
        return bool(frame[idx, 3] > 0 and frame[idx, 2] >= CONF_THRESH)
    xy = frame[idx, :2]
    return bool((xy != 0).any() and 0 <= xy[0] <= 1 and 0 <= xy[1] <= 1)


def _xy(frame: np.ndarray, idx: int) -> np.ndarray | None:
    if not _valid_joint(frame, idx):
        return None
    return np.array([float(frame[idx, 0]), float(frame[idx, 1])], dtype=np.float32)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _joint_angle(hand: np.ndarray, a: int, b: int, c: int) -> float:
    pa, pb, pc = _xy(hand, a), _xy(hand, b), _xy(hand, c)
    if pa is None or pb is None or pc is None:
        return float("nan")
    return _angle_between(pa - pb, pc - pb)


def _normalize_hand(hand: np.ndarray) -> np.ndarray:
    """Wrist-centered, scaled by wrist-to-middle-MCP distance."""
    out = hand.copy()
    wrist = _xy(hand, HAND_WRIST)
    mid = _xy(hand, HAND_MIDDLE_MCP)
    if wrist is None or mid is None:
        return out
    span = float(np.linalg.norm(mid - wrist))
    if span < 1e-6:
        return out
    for i in range(len(out)):
        if _valid_joint(hand, i):
            out[i, 0] = (hand[i, 0] - wrist[0]) / span
            out[i, 1] = (hand[i, 1] - wrist[1]) / span
    return out


def _palm_normal(hand: np.ndarray) -> np.ndarray:
    w = _xy(hand, HAND_WRIST)
    idx_mcp = _xy(hand, HAND_INDEX_MCP)
    pinky_mcp = _xy(hand, HAND_PINKY_MCP)
    if w is None or idx_mcp is None or pinky_mcp is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([idx_mcp[0] - w[0], idx_mcp[1] - w[1], 0.0], dtype=np.float32)
    v2 = np.array([pinky_mcp[0] - w[0], pinky_mcp[1] - w[1], 0.0], dtype=np.float32)
    n = np.cross(v1, v2)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return (n / norm).astype(np.float32)


def _hand_validity(hand: np.ndarray) -> float:
    if len(hand) == 0:
        return 0.0
    valid = sum(1 for i in range(len(hand)) if _valid_joint(hand, i))
    return valid / max(len(hand), 1)


def pick_active_hand(wholebody: np.ndarray) -> Literal["left", "right"]:
    lh = wholebody[LEFT_HAND_SLICE]
    rh = wholebody[RIGHT_HAND_SLICE]
    lv, rv = _hand_validity(lh), _hand_validity(rh)
    return "left" if lv >= rv else "right"


def _hand_slice(wholebody: np.ndarray, side: Literal["left", "right"]) -> np.ndarray:
    return wholebody[LEFT_HAND_SLICE] if side == "left" else wholebody[RIGHT_HAND_SLICE]


def _body_wrist_elbow(body: np.ndarray, side: Literal["left", "right"]) -> tuple[np.ndarray | None, np.ndarray | None]:
    elbow_idx = BODY_L_ELBOW if side == "left" else BODY_R_ELBOW
    wrist_idx = BODY_L_WRIST if side == "left" else BODY_R_WRIST
    return _xy(body, elbow_idx), _xy(body, wrist_idx)


def _wrist_flexion_deg(body: np.ndarray, hand: np.ndarray, side: Literal["left", "right"]) -> float:
    elbow, body_wrist = _body_wrist_elbow(body, side)
    hand_wrist = _xy(hand, HAND_WRIST)
    mid_mcp = _xy(hand, HAND_MIDDLE_MCP)
    if elbow is None or body_wrist is None or hand_wrist is None or mid_mcp is None:
        return float("nan")
    forearm = body_wrist - elbow
    hand_axis = mid_mcp - hand_wrist
    return _angle_between(forearm, hand_axis)


def _finger_curls(hand: np.ndarray) -> dict[str, list[float]]:
    curls: dict[str, list[float]] = {}
    for finger, triples in FINGER_CURL_JOINTS.items():
        angles = [_joint_angle(hand, a, b, c) for a, b, c in triples]
        curls[finger] = [float(a) if not np.isnan(a) else 0.0 for a in angles]
    return curls


def frame_confidence(body: np.ndarray, hand: np.ndarray, side: Literal["left", "right"]) -> float:
    elbow, wrist = _body_wrist_elbow(body, side)
    scores = [_hand_validity(hand)]
    if elbow is not None:
        scores.append(1.0)
    if wrist is not None:
        scores.append(1.0)
    return float(np.mean(scores)) if scores else 0.0


def extract_frame_features(
    wholebody: np.ndarray,
    *,
    active_hand: Literal["left", "right"] | None = None,
    timestamp_ms: int = 0,
) -> dict[str, Any]:
    """Extract one feature dict from a ``(133, 4)`` wholebody frame."""
    side = active_hand or pick_active_hand(wholebody)
    body = wholebody[BODY_SLICE]
    hand = _normalize_hand(_hand_slice(wholebody, side))
    conf = frame_confidence(body, hand, side)
    palm = _palm_normal(hand)
    flexion = _wrist_flexion_deg(body, hand, side)
    return {
        "palm_normal": palm.tolist(),
        "finger_curls": _finger_curls(hand),
        "wrist_flexion_deg": float(flexion) if not np.isnan(flexion) else 0.0,
        "confidence": conf,
        "timestamp_ms": timestamp_ms,
        "active_hand": side,
    }


def sequence_from_wholebody(
    seq: np.ndarray,
    fps: float = 25.0,
    *,
    active_hand: Literal["left", "right"] | None = None,
) -> tuple[list[dict[str, Any]], Literal["left", "right"]]:
    """Extract feature sequence from ``(T, 133, 4)`` array."""
    if seq.ndim != 3 or seq.shape[1] < 133:
        raise ValueError(f"Expected (T, 133, 4) wholebody sequence, got {seq.shape}")

    # Pick dominant hand from first valid frame or majority vote
    if active_hand is None:
        votes: list[str] = []
        for t in range(min(seq.shape[0], 10)):
            votes.append(pick_active_hand(seq[t]))
        active_hand = max(set(votes), key=votes.count)  # type: ignore[arg-type]

    features: list[dict[str, Any]] = []
    for t in range(seq.shape[0]):
        ts = int(t * 1000.0 / max(fps, 1.0))
        features.append(extract_frame_features(seq[t], active_hand=active_hand, timestamp_ms=ts))
    return features, active_hand


def extract_sequence_features(
    seq: np.ndarray,
    fps: float = 25.0,
    *,
    active_hand: Literal["left", "right"] | None = None,
) -> list[dict[str, Any]]:
    feats, _ = sequence_from_wholebody(seq, fps, active_hand=active_hand)
    return feats


def palm_normal_angle_deg(a: list[float], b: list[float]) -> float:
    """Angle between two palm normal vectors in degrees."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos_a = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def palm_facing_direction(ref_z: float, user_z: float) -> str:
    if user_z * ref_z >= 0:
        if abs(user_z) < abs(ref_z) * 0.7:
            return "palm not facing the camera enough"
        return "palm orientation close to reference"
    return "palm facing the wrong way (toward or away from camera)"
