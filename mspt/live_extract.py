"""MediaPipe extraction for MSPT — IMAGE mode (webcam) and VIDEO mode (file clips)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import slr_common as C  # noqa: E402
from face_landmarks import FACE_IDXS, NUM_FACE  # noqa: E402
from mspt.pose_utils import (  # noqa: E402
    NUM_HAND,
    NUM_POSE,
    body_from_pose,
    landmarks_from_results,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


from mspt.mediapipe_models import resolve_models_dir  # noqa: E402


def _resolve_models_dir(models_dir: Path | None) -> Path:
    return resolve_models_dir(models_dir)


def _empty_face() -> np.ndarray:
    return np.zeros((NUM_FACE, 4), dtype=np.float32)


def face_from_result(face_result) -> np.ndarray:
    out = _empty_face()
    if not face_result or not face_result.face_landmarks:
        return out
    lms = face_result.face_landmarks[0]
    for j, idx in enumerate(FACE_IDXS):
        if idx < len(lms):
            lm = lms[idx]
            out[j] = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
    return out


class LiveStreamExtractor:
    """Pose + hands + face from a single BGR frame (matches training caches)."""

    def __init__(self, models_dir: Path | None = None, frame_size: int = 224):
        models_dir = _resolve_models_dir(models_dir)
        pose_path = models_dir / "pose_landmarker_full.task"
        hand_path = models_dir / "hand_landmarker.task"
        face_path = models_dir / "face_landmarker.task"
        missing = [p for p in (pose_path, hand_path, face_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing MediaPipe model(s):\n"
                + "\n".join(f"  - {p}" for p in missing)
                + f"\nSet INCLUDE_ML_ROOT to the folder containing models/ "
                f"(expected under {C.INCLUDE_ML_ROOT / 'models'})."
            )

        self.frame_size = frame_size
        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
        hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        face_opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        self.pose_lm = PoseLandmarker.create_from_options(pose_opts)
        self.hand_lm = HandLandmarker.create_from_options(hand_opts)
        self.face_lm = FaceLandmarker.create_from_options(face_opts)

    def close(self) -> None:
        self.pose_lm.close()
        self.hand_lm.close()
        self.face_lm.close()

    def __enter__(self) -> LiveStreamExtractor:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def process_frame(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return hand ``(42,2)``, body ``(33,2)``, face ``(NUM_FACE,2)``."""
        small = cv2.resize(bgr, (self.frame_size, self.frame_size))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_r = self.pose_lm.detect(mp_img)
        hand_r = self.hand_lm.detect(mp_img)
        face_r = self.face_lm.detect(mp_img)

        lm54 = landmarks_from_results(pose_r, hand_r)
        hands = lm54[NUM_POSE : NUM_POSE + 2 * NUM_HAND, :2].astype(np.float32)
        body = body_from_pose(pose_r)[:, :2].astype(np.float32)
        face = face_from_result(face_r)[:, :2].astype(np.float32)
        return hands, body, face


class VideoClipExtractor:
    """VIDEO-mode extraction aligned with training caches (detect_for_video + timestamps)."""

    def __init__(self, models_dir: Path | None = None, frame_size: int = 224):
        models_dir = _resolve_models_dir(models_dir)
        pose_path = models_dir / "pose_landmarker_full.task"
        hand_path = models_dir / "hand_landmarker.task"
        face_path = models_dir / "face_landmarker.task"
        missing = [p for p in (pose_path, hand_path, face_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing MediaPipe model(s):\n" + "\n".join(f"  - {p}" for p in missing)
            )
        self.frame_size = frame_size
        self.pose_lm = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1,
            )
        )
        self.hand_lm = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )
        )
        self.face_lm = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(face_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
            )
        )

    def close(self) -> None:
        self.pose_lm.close()
        self.hand_lm.close()
        self.face_lm.close()

    def __enter__(self) -> VideoClipExtractor:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def extract_clip(
        self,
        video_path: str | Path,
        frame_stride: int = 1,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Return per-frame (hands, body, face) lists using VIDEO running mode."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], [], []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ts_ms = 0
        frame_i = 0
        hands_seq: list[np.ndarray] = []
        body_seq: list[np.ndarray] = []
        face_seq: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_stride > 1 and (frame_i % frame_stride) != 0:
                    frame_i += 1
                    continue
                frame_i += 1
                small = cv2.resize(frame, (self.frame_size, self.frame_size))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                if hands_seq:
                    ts_ms += max(1, int(1000 / fps)) * frame_stride
                pose_r = self.pose_lm.detect_for_video(mp_img, ts_ms)
                hand_r = self.hand_lm.detect_for_video(mp_img, ts_ms)
                face_r = self.face_lm.detect_for_video(mp_img, ts_ms)
                lm54 = landmarks_from_results(pose_r, hand_r)
                hands_seq.append(
                    lm54[NUM_POSE : NUM_POSE + 2 * NUM_HAND, :2].astype(np.float32)
                )
                body_seq.append(body_from_pose(pose_r)[:, :2].astype(np.float32))
                face_seq.append(face_from_result(face_r)[:, :2].astype(np.float32))
        finally:
            cap.release()
        return hands_seq, body_seq, face_seq
