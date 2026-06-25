"""Extract RTMLIB wholebody keypoints from uploaded video."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dashboard.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mspt.rtmlib_preprocess import RtmlibWholebodyExtractor  # noqa: E402

MAX_VIDEO_SEC = 10.0
MAX_POSE_WIDTH = 960
ALLOWED_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def _resize_for_pose(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / w
    resized = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return resized, scale, scale


class VideoOrientationExtractor:
    """Lazy-loaded RTMLIB extractor for orientation analysis."""

    def __init__(self) -> None:
        self._extractor: RtmlibWholebodyExtractor | None = None

    def _get_extractor(self) -> RtmlibWholebodyExtractor:
        if self._extractor is None:
            self._extractor = RtmlibWholebodyExtractor()
        return self._extractor

    def extract_from_path(self, video_path: Path) -> tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        max_frames = int(MAX_VIDEO_SEC * fps)
        extractor = self._get_extractor()
        frames: list[np.ndarray] = []

        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            pose_frame, _, _ = _resize_for_pose(frame, MAX_POSE_WIDTH)
            wb = extractor.process_frame(pose_frame)
            frames.append(wb)

        cap.release()
        if not frames:
            raise ValueError("No frames decoded from video")
        return np.stack(frames, axis=0).astype(np.float32), fps

    def extract_from_bytes(self, data: bytes, suffix: str = ".mp4") -> tuple[np.ndarray, float]:
        suffix = suffix.lower() if suffix.lower() in ALLOWED_SUFFIXES else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            return self.extract_from_path(Path(tmp.name))


_extractor_singleton: VideoOrientationExtractor | None = None


def reset_video_extractor() -> None:
    """Drop cached extractor (e.g. after code hot-reload)."""
    global _extractor_singleton
    if _extractor_singleton is not None and _extractor_singleton._extractor is not None:
        try:
            _extractor_singleton._extractor.close()
        except Exception:
            pass
    _extractor_singleton = None


def get_video_extractor() -> VideoOrientationExtractor:
    global _extractor_singleton
    if _extractor_singleton is None:
        _extractor_singleton = VideoOrientationExtractor()
    return _extractor_singleton
