#!/usr/bin/env python3
"""Extract face landmark sidecars (T, NUM_FACE, 4) aligned with include50_lab manifests."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

from face_landmarks import FACE_IDXS, NUM_FACE  # noqa: E402

import slr_common as C  # noqa: E402

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_face_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "face_landmarker.task"
    if not path.is_file():
        import urllib.request

        print("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
    return path


def _empty_frame() -> np.ndarray:
    return np.zeros((NUM_FACE, 4), dtype=np.float32)


def face_from_result(face_result) -> np.ndarray:
    """(NUM_FACE, 4) with x,y,z,visibility in normalized image coords."""
    out = _empty_frame()
    if not face_result or not face_result.face_landmarks:
        return out
    lms = face_result.face_landmarks[0]
    for j, idx in enumerate(FACE_IDXS):
        if idx < len(lms):
            lm = lms[idx]
            out[j] = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
    return out


def _encode_frames(
    frames: list,
    face_lm: FaceLandmarker,
    fps: float,
    frame_size: int,
    ts_start_ms: int,
) -> tuple[list[np.ndarray], int]:
    """Returns encoded frames and the next valid timestamp for VIDEO mode."""
    seq: list[np.ndarray] = []
    ts = ts_start_ms
    for i, frame in enumerate(frames):
        small = cv2.resize(frame, (frame_size, frame_size))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if i > 0:
            ts += max(1, int(1000 / fps))
        fr = face_lm.detect_for_video(mp_img, ts)
        seq.append(face_from_result(fr))
    return seq, ts + 1


def extract_face_video(
    video_path: str, out_path: Path, face_lm: FaceLandmarker | None = None, frame_size: int = 224
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return 0

    own_lm = False
    if face_lm is None:
        model_path = ensure_face_model(C.INCLUDE_ML_ROOT / "models")
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
        )
        face_lm = FaceLandmarker.create_from_options(opts)
        own_lm = True
    try:
        seq, _ = _encode_frames(frames, face_lm, fps, frame_size, ts_start_ms=0)
    finally:
        if own_lm:
            face_lm.close()

    if not seq:
        return 0
    n_frames = len(seq)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.stack(seq, axis=0).astype(np.float32))
    del frames, seq
    gc.collect()
    return n_frames


def process_manifest(manifest_csv: Path, out_dir: Path, force: bool = False) -> int:
    df = pd.read_csv(manifest_csv)
    n_ok = 0
    model_path = ensure_face_model(C.INCLUDE_ML_ROOT / "models")
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    face_lm = FaceLandmarker.create_from_options(opts)
    ts_cursor = 0
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=manifest_csv.name):
            video = C.resolve_video_path(row["path"])
            stem = Path(row["path"]).stem
            out_path = out_dir / row["label"] / f"{stem}.npy"
            if out_path.exists() and not force:
                n_ok += 1
                continue
            if not video.is_file():
                continue
            ts_cursor = _extract_face_video_with_lm(str(video), out_path, face_lm, ts_cursor)
            if out_path.exists():
                n_ok += 1
    finally:
        face_lm.close()
    return n_ok


def _extract_face_video_with_lm(
    video_path: str, out_path: Path, face_lm: FaceLandmarker, ts_start_ms: int, frame_size: int = 224
) -> int:
    """Encode one clip; returns next monotonic timestamp for the shared landmarker."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ts_start_ms
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return ts_start_ms
    seq, ts_next = _encode_frames(frames, face_lm, fps, frame_size, ts_start_ms)
    if not seq:
        return ts_start_ms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.stack(seq, axis=0).astype(np.float32))
    del frames, seq
    gc.collect()
    return ts_next


def face_ready(lab_root: Path, min_fraction: float = 0.9) -> bool:
    lm = lab_root / "cache" / "landmarks"
    face = lab_root / "cache" / "landmarks_face"
    n_lm = sum(1 for _ in lm.rglob("*.npy")) if lm.is_dir() else 0
    n_face = sum(1 for _ in face.rglob("*.npy")) if face.is_dir() else 0
    if n_lm == 0:
        return n_face > 0
    return n_face >= min_fraction * n_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    args = ap.parse_args()
    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    lab_root = C._resolve_lab_root()
    out_dir = lab_root / "cache" / "landmarks_face"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    total = 0
    for split in splits:
        mp = lab_root / "manifests" / f"{split}.csv"
        if mp.exists():
            total += process_manifest(mp, out_dir, force=args.force)
    print(f"face landmarks -> {out_dir} ({total} clips processed or present)")


if __name__ == "__main__":
    main()
