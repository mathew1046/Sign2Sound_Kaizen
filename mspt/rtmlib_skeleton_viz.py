"""Skeleton overlay for rtmlib COCO-WholeBody (133 keypoints)."""

from __future__ import annotations

import cv2
import numpy as np

from mspt.rtmlib_preprocess import (
    BODY_SLICE,
    FACE_SLICE,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
)
from mspt.skeleton_viz import (
    HAND_CONNECTIONS,
    _draw_connections,
    _fit_points,
    _map_pt,
    _valid_pt,
)

# COCO-17 body topology
COCO_BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


def _xy_from_wholebody(wb: np.ndarray) -> np.ndarray:
    """``(133, 4)`` -> ``(133, 2)`` with invalid joints zeroed."""
    xy = np.array(wb[..., :2], dtype=np.float32, copy=True)
    if wb.shape[-1] >= 4:
        valid = wb[..., 3] > 0
        xy[~valid] = 0.0
    return xy


def wholebody_frame_to_streams(wb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single frame ``(133, 4)`` -> hands (42,2), body (17,2), face (68,2)."""
    xy = _xy_from_wholebody(wb)
    lh = xy[LEFT_HAND_SLICE]
    rh = xy[RIGHT_HAND_SLICE]
    hands = np.concatenate([lh, rh], axis=0)
    body = xy[BODY_SLICE]
    face = xy[FACE_SLICE]
    return hands, body, face


def render_rtmlib_skeleton_panel(
    wholebody_frame: np.ndarray | None = None,
    *,
    hands: np.ndarray | None = None,
    body: np.ndarray | None = None,
    face: np.ndarray | None = None,
    panel_size: int = 480,
) -> np.ndarray:
    """Render BGR skeleton panel from rtmlib wholebody or stream arrays."""
    if wholebody_frame is not None:
        hands, body, face = wholebody_frame_to_streams(wholebody_frame)

    hands = np.asarray(hands, dtype=np.float32)
    body = np.asarray(body, dtype=np.float32)
    face = np.asarray(face, dtype=np.float32) if face is not None else np.zeros((0, 2), dtype=np.float32)

    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    panel[:] = (28, 28, 32)

    lh = hands[:21] if hands.shape[0] >= 42 else hands[: min(21, len(hands))]
    rh = hands[21:42] if hands.shape[0] >= 42 else np.zeros((21, 2), dtype=np.float32)

    scale, ox, oy = _fit_points([body, lh, rh, face] if face.size else [body, lh, rh], panel_size, panel_size)

    lt_body = max(2, panel_size // 120)
    lt_hand = max(2, panel_size // 100)
    jr_body = max(3, panel_size // 80)
    jr_hand = max(3, panel_size // 70)

    _draw_connections(
        panel, body, COCO_BODY_CONNECTIONS, scale, ox, oy,
        line_color=(80, 255, 120),
        joint_color=(200, 255, 200),
        line_thickness=lt_body,
        joint_radius=jr_body,
    )
    _draw_connections(
        panel, lh, HAND_CONNECTIONS, scale, ox, oy,
        line_color=(250, 44, 121),
        joint_color=(255, 180, 220),
        line_thickness=lt_hand,
        joint_radius=jr_hand,
    )
    _draw_connections(
        panel, rh, HAND_CONNECTIONS, scale, ox, oy,
        line_color=(66, 117, 245),
        joint_color=(180, 210, 255),
        line_thickness=lt_hand,
        joint_radius=jr_hand,
    )

    if face.size:
        for p in face:
            if _valid_pt(p):
                pt = _map_pt(p, scale, ox, oy)
                if pt:
                    cv2.circle(panel, pt, max(1, jr_hand // 2), (140, 140, 160), -1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (panel_size - 1, panel_size - 1), (90, 90, 100), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "rtmlib skeleton",
        (10, panel_size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 190),
        2,
        cv2.LINE_AA,
    )
    return panel
