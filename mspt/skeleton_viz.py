"""Draw MediaPipe-style skeleton panels for live webcam overlay."""

from __future__ import annotations

import cv2
import numpy as np

# BlazePose 33-landmark topology
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (12, 24),
    (24, 23), (23, 11), (15, 17), (15, 19), (15, 21), (16, 18),
    (16, 20), (16, 22), (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _valid_pt(xy: np.ndarray) -> bool:
    return bool((xy != 0).any() and 0 <= xy[0] <= 1 and 0 <= xy[1] <= 1)


def _fit_points(
    point_sets: list[np.ndarray],
    panel_w: int,
    panel_h: int,
    margin_frac: float = 0.08,
) -> tuple[float, float, float]:
    """Return scale and offset (ox, oy) mapping normalized xy -> panel pixels."""
    pts = []
    for arr in point_sets:
        for p in arr:
            if _valid_pt(p):
                pts.append(p)
    if not pts:
        return 1.0, panel_w * 0.5, panel_h * 0.5
    pts = np.array(pts, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 0.05)
    margin = margin_frac * min(panel_w, panel_h)
    scale = (min(panel_w, panel_h) - 2 * margin) / span
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    ox = panel_w * 0.5 - cx * scale
    oy = panel_h * 0.5 - cy * scale
    return scale, ox, oy


def _map_pt(xy: np.ndarray, scale: float, ox: float, oy: float) -> tuple[int, int] | None:
    if not _valid_pt(xy):
        return None
    x = int(xy[0] * scale + ox)
    y = int(xy[1] * scale + oy)
    return x, y


def _draw_connections(
    canvas: np.ndarray,
    joints: np.ndarray,
    connections: tuple[tuple[int, int], ...],
    scale: float,
    ox: float,
    oy: float,
    line_color: tuple[int, int, int],
    joint_color: tuple[int, int, int],
    line_thickness: int,
    joint_radius: int,
) -> None:
    for a, b in connections:
        if a >= len(joints) or b >= len(joints):
            continue
        pa = _map_pt(joints[a], scale, ox, oy)
        pb = _map_pt(joints[b], scale, ox, oy)
        if pa and pb:
            cv2.line(canvas, pa, pb, line_color, line_thickness, cv2.LINE_AA)
    for j in range(len(joints)):
        p = _map_pt(joints[j], scale, ox, oy)
        if p:
            cv2.circle(canvas, p, joint_radius, joint_color, -1, cv2.LINE_AA)


def render_skeleton_panel(
    hands: np.ndarray,
    body: np.ndarray,
    face: np.ndarray | None = None,
    panel_size: int = 400,
) -> np.ndarray:
    """Render a BGR skeleton canvas ``(panel_size, panel_size, 3)``."""
    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    panel[:] = (28, 28, 32)

    lh = hands[:21] if hands.shape[0] >= 42 else hands[: min(21, len(hands))]
    rh = hands[21:42] if hands.shape[0] >= 42 else np.zeros((21, 2), dtype=np.float32)

    scale, ox, oy = _fit_points([body, lh, rh], panel_size, panel_size)

    lt_body = max(2, panel_size // 120)
    lt_hand = max(2, panel_size // 100)
    jr_body = max(3, panel_size // 80)
    jr_hand = max(3, panel_size // 70)

    _draw_connections(
        panel, body, POSE_CONNECTIONS, scale, ox, oy,
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

    if face is not None and face.shape[0] > 0:
        for p in face:
            pt = _map_pt(p, scale, ox, oy)
            if pt:
                cv2.circle(panel, pt, max(1, jr_hand // 2), (140, 140, 160), -1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (panel_size - 1, panel_size - 1), (90, 90, 100), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "skeleton",
        (10, panel_size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def composite_bottom_right(
    frame: np.ndarray,
    panel: np.ndarray,
    margin: int = 16,
) -> None:
    """Paste ``panel`` onto ``frame`` at bottom-right (in-place)."""
    _composite_corner(frame, panel, margin, corner="br")


def composite_bottom_left(
    frame: np.ndarray,
    panel: np.ndarray,
    margin: int = 16,
) -> None:
    """Paste ``panel`` onto ``frame`` at bottom-left (in-place)."""
    _composite_corner(frame, panel, margin, corner="bl")


def _composite_corner(
    frame: np.ndarray,
    panel: np.ndarray,
    margin: int,
    corner: str,
) -> None:
    fh, fw = frame.shape[:2]
    ph, pw = panel.shape[:2]
    if corner == "bl":
        x0 = margin
    else:
        x0 = max(0, fw - pw - margin)
    y0 = max(0, fh - ph - margin)
    x1, y1 = x0 + pw, y0 + ph
    if x1 > fw or y1 > fh:
        panel = cv2.resize(panel, (min(pw, fw - margin), min(ph, fh - margin)))
        ph, pw = panel.shape[:2]
        if corner == "bl":
            x0 = margin
        else:
            x0 = fw - pw - margin
        y0 = fh - ph - margin
        x1, y1 = x0 + pw, y0 + ph

    roi = frame[y0:y1, x0:x1]
    alpha = 0.92
    cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0, roi)
