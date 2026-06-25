"""Stitch gloss sequences with trim, hold, crossfade, and rtmlib wholebody bridges."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashboard.config import (
    BRIDGE_FRAMES,
    CROSSFADE_FRAMES,
    FRAME_SIZE,
    FPS,
    HOLD_FRAMES,
    WHOLEBODY_DIR,
)
from mspt.rtmlib_skeleton_viz import render_rtmlib_skeleton_panel


def _is_empty_frame(frame: np.ndarray, thresh: float = 0.02) -> bool:
    return float(np.mean(frame > 8)) < thresh


def trim_wholebody(seq: np.ndarray) -> np.ndarray:
    """Drop leading/trailing frames with no visible keypoints."""
    if seq.shape[0] == 0:
        return seq
    valid = seq[..., 3] > 0 if seq.shape[-1] >= 4 else np.any(seq[..., :2] > 0.01, axis=-1)
    frame_ok = np.any(valid, axis=1) if valid.ndim == 2 else valid
    idx = np.flatnonzero(frame_ok)
    if idx.size == 0:
        return seq
    return seq[idx[0] : idx[-1] + 1]


def render_wholebody_frame(wb: np.ndarray, panel_size: int = FRAME_SIZE) -> np.ndarray:
    return render_rtmlib_skeleton_panel(wb, panel_size=panel_size)


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    if n <= 0:
        return []
    out = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        blended = ((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32)).astype(np.uint8)
        out.append(blended)
    return out


def load_exemplar_wholebody(gloss: str, exemplar_id: str, root: Path) -> np.ndarray:
    path = root / gloss / f"{exemplar_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing wholebody cache: {path}")
    return trim_wholebody(np.load(path))


def _interp_wholebody(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = t * t * (3.0 - 2.0 * t)
    out = (1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32)
    return out.astype(np.float32)


def bridge_between(
    sk_a: np.ndarray,
    sk_b: np.ndarray,
) -> list[np.ndarray]:
    """Bridge two wholebody sequences with smooth keypoint interpolation."""
    n = BRIDGE_FRAMES
    if sk_a.shape[0] == 0 or sk_b.shape[0] == 0:
        return []
    end_a = np.asarray(sk_a[-1], dtype=np.float32)
    start_b = np.asarray(sk_b[0], dtype=np.float32)
    frames: list[np.ndarray] = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        mid = _interp_wholebody(end_a, start_b, t)
        frames.append(render_wholebody_frame(mid))
    if not frames and sk_a.shape[0] and sk_b.shape[0]:
        img_a = render_wholebody_frame(sk_a[-1])
        img_b = render_wholebody_frame(sk_b[0])
        hold = [img_a] * HOLD_FRAMES
        return hold + crossfade(img_a, img_b, CROSSFADE_FRAMES)
    return frames


def compose_glosses(
    gloss_entries: list[tuple[str, str]],
    wholebody_dir: Path | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """gloss_entries: list of (gloss, exemplar_id). Returns PNG frames + segments."""
    root = wholebody_dir or WHOLEBODY_DIR
    frames: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    cached_sk: list[np.ndarray] = []

    for i, (gloss, ex_id) in enumerate(gloss_entries):
        sk = load_exemplar_wholebody(gloss, ex_id, root)
        cached_sk.append(sk)
        start_idx = len(frames)
        for t in range(sk.shape[0]):
            frames.append(render_wholebody_frame(sk[t]))
        segments.append(
            {
                "gloss": gloss,
                "exemplar_id": ex_id,
                "start_frame": start_idx,
                "end_frame": len(frames),
                "num_frames": int(sk.shape[0]),
            }
        )
        if i + 1 < len(gloss_entries):
            sk2 = load_exemplar_wholebody(gloss_entries[i + 1][0], gloss_entries[i + 1][1], root)
            bridge = bridge_between(sk, sk2)
            frames.extend(bridge)

    return frames, segments


def frames_to_timeline_payload(
    frames: list[np.ndarray],
    segments: list[dict],
    encode_b64: bool = True,
) -> dict:
    encoded = []
    for i, fr in enumerate(frames):
        item: dict[str, Any] = {"t": i, "index": i}
        if encode_b64:
            _, buf = cv2.imencode(".png", fr)
            item["frame_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        encoded.append(item)
    return {
        "fps": FPS,
        "frame_size": FRAME_SIZE,
        "num_frames": len(frames),
        "segments": segments,
        "frames": encoded,
    }


def resolve_gloss_entries(
    glosses: list[str],
    catalog: dict,
) -> list[tuple[str, str]]:
    gloss_map = {g["gloss"]: g for g in catalog["glosses"]}
    entries = []
    missing = []
    for g in glosses:
        entry = gloss_map.get(g)
        if not entry or not entry.get("default_exemplar_id"):
            missing.append(g)
            continue
        entries.append((g, entry["default_exemplar_id"]))
    if missing:
        raise ValueError(f"No exemplar for glosses: {', '.join(missing)}")
    return entries
