"""Hand bbox crop + MediaPipe landmark extraction with full-frame remap."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hand_detect.cansik_yolo import CansikHandDetector, expand_crop  # noqa: E402

DEFAULT_CANSIK_CFG = REPO_ROOT / "weights" / "cansik" / "cross-hands-yolov4-tiny.cfg"
DEFAULT_CANSIK_WEIGHTS = REPO_ROOT / "weights" / "cansik" / "cross-hands-yolov4-tiny.weights"
CANSIK_WEIGHTS_URL = (
    "https://github.com/cansik/yolo-hand-detection/raw/master/models/"
    "cross-hands-yolov4-tiny.weights"
)

_warned_missing_weights = False


def default_cansik_paths() -> tuple[Path, Path]:
    return DEFAULT_CANSIK_CFG, DEFAULT_CANSIK_WEIGHTS


def cansik_weights_available(cfg: Path | None = None, weights: Path | None = None) -> bool:
    cfg_path, weights_path = cfg or DEFAULT_CANSIK_CFG, weights or DEFAULT_CANSIK_WEIGHTS
    return cfg_path.is_file() and weights_path.is_file()


def union_hand_bbox(
    detections: list[tuple[float, int, int, int, int]],
    frame_w: int,
    frame_h: int,
    pad_frac: float = 0.15,
    min_size: int = 12,
) -> tuple[int, int, int, int] | None:
    """Union of hand bboxes with padding; returns (x0, y0, w, h) or None."""
    if not detections:
        return None
    x_min = min(d[1] for d in detections)
    y_min = min(d[2] for d in detections)
    x_max = max(d[1] + d[3] for d in detections)
    y_max = max(d[2] + d[4] for d in detections)
    w = x_max - x_min
    h = y_max - y_min
    if w < min_size or h < min_size:
        return None
    return expand_crop(x_min, y_min, w, h, frame_w, frame_h, pad_frac=pad_frac)


def crop_frame(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, w, h = bbox
    crop = frame[y0 : y0 + h, x0 : x0 + w]
    return crop, (x0, y0, w, h)


def _landmarks_from_results(results) -> np.ndarray:
    """MediaPipe HandLandmarker results -> (42, 3) in input-image normalized coords."""
    frame_landmarks = np.zeros((42, 3), dtype=np.float32)
    if not results.hand_landmarks:
        return frame_landmarks
    for hand_idx, hand_lms in enumerate(results.hand_landmarks):
        hand_label = results.handedness[hand_idx][0].category_name
        offset = 0 if hand_label == "Left" else 21
        for i, lm in enumerate(hand_lms):
            if i >= 21:
                break
            frame_landmarks[offset + i] = [lm.x, lm.y, lm.z]
    return frame_landmarks


def remap_landmarks_to_full_frame(
    landmarks: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
) -> np.ndarray:
    """Remap crop-normalized landmarks to full-frame normalized [0, 1] coords."""
    x0, y0, crop_w, crop_h = crop_rect
    frame_w, frame_h = frame_wh
    out = landmarks.copy()
    if crop_w <= 0 or crop_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return out
    for i in range(len(out)):
        x_crop, y_crop, z = out[i]
        out[i, 0] = (x_crop * crop_w + x0) / frame_w
        out[i, 1] = (y_crop * crop_h + y0) / frame_h
        out[i, 2] = z
    return out


def detect_on_frame(
    landmarker,
    frame_bgr: np.ndarray,
    *,
    use_crop: bool = True,
    detector: CansikHandDetector | None = None,
    pad_frac: float = 0.15,
) -> np.ndarray:
    """Run MediaPipe on crop (if detector finds hands) else full frame; return (42, 3)."""
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    crop_rect: tuple[int, int, int, int] | None = None
    input_rgb = frame_rgb

    if use_crop and detector is not None:
        dets, _ = detector.detect(frame_bgr)
        crop_rect = union_hand_bbox(dets, frame_w, frame_h, pad_frac=pad_frac)
        if crop_rect is not None:
            crop_bgr, crop_rect = crop_frame(frame_bgr, crop_rect)
            if crop_bgr.size > 0:
                input_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_rgb)
    results = landmarker.detect(mp_image)
    landmarks = _landmarks_from_results(results)

    if crop_rect is not None:
        landmarks = remap_landmarks_to_full_frame(landmarks, crop_rect, (frame_w, frame_h))
    return landmarks


def try_create_cansik_detector(
    cfg: Path | None = None,
    weights: Path | None = None,
    confidence: float = 0.2,
) -> CansikHandDetector | None:
    """Return CansikHandDetector or None if weights missing (logs once)."""
    global _warned_missing_weights
    cfg_path, weights_path = cfg or DEFAULT_CANSIK_CFG, weights or DEFAULT_CANSIK_WEIGHTS
    if not cansik_weights_available(cfg_path, weights_path):
        if not _warned_missing_weights:
            print(
                f"[hand_crop] Cansik weights not found at {weights_path}; "
                f"download from {CANSIK_WEIGHTS_URL} — using full-frame MediaPipe."
            )
            _warned_missing_weights = True
        return None
    return CansikHandDetector(cfg_path, weights_path, confidence=confidence)


def extract_frame_landmarks(
    landmarker,
    frame_bgr: np.ndarray,
    *,
    use_crop: bool = True,
    detector: CansikHandDetector | None = None,
    pad_frac: float = 0.15,
) -> np.ndarray:
    """Public API: cropped or full-frame landmark extraction -> (42, 3)."""
    return detect_on_frame(
        landmarker,
        frame_bgr,
        use_crop=use_crop,
        detector=detector,
        pad_frac=pad_frac,
    )
