"""Extract full 33-pose body landmarks — one video / frame at a time, no RAM hoarding."""

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

NOTEBOOKS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NOTEBOOKS))
import slr_common as C  # noqa: E402

from mspt.pose_utils import NUM_BODY, body_from_pose, body_ready  # noqa: E402

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_body_video(
    video_path: str,
    out_path: Path,
    pose_lm: PoseLandmarker,
    frame_size: int = 224,
    frame_stride: int = 1,
) -> int:
    """Stream frames from disk; write .npy once at end. Never hold full video in RAM."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seq: list[np.ndarray] = []
    ts = 0
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (frame_i % frame_stride) != 0:
                frame_i += 1
                continue
            small = cv2.resize(frame, (frame_size, frame_size))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if seq:
                ts += max(1, int(1000 / fps)) * frame_stride
            pr = pose_lm.detect_for_video(mp_img, ts)
            seq.append(body_from_pose(pr))
            del frame, small, rgb, mp_img, pr
            frame_i += 1
    finally:
        cap.release()

    if not seq:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.stack(seq, axis=0).astype(np.float32))
    n = len(seq)
    del seq
    gc.collect()
    return n


def process_manifest(
    manifest_csv: Path,
    out_dir: Path,
    pose_model: Path,
    force: bool = False,
    frame_stride: int = 1,
    log_path: Path | None = None,
) -> dict[str, int]:
    df = pd.read_csv(manifest_csv)
    stats = {"ok": 0, "skipped_existing": 0, "missing_video": 0, "failed": 0}
    failures: list[str] = []
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )
    pose_lm = PoseLandmarker.create_from_options(opts)
    ts_cursor = 0
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=manifest_csv.name):
            stem = Path(row["path"]).stem
            out_path = out_dir / row["label"] / f"{stem}.npy"
            if out_path.exists() and not force:
                stats["skipped_existing"] += 1
                stats["ok"] += 1
                continue
            video = C.resolve_video_path(row["path"])
            if not video.is_file():
                stats["missing_video"] += 1
                failures.append(f"missing_video,{video}")
                continue
            try:
                ts_cursor = _extract_body_video_with_lm(
                    str(video), out_path, pose_lm, ts_cursor, frame_stride
                )
                if out_path.exists():
                    stats["ok"] += 1
                else:
                    stats["failed"] += 1
                    failures.append(f"empty,{video}")
            except Exception as exc:
                stats["failed"] += 1
                failures.append(f"error,{video},{exc}")
                tqdm.write(f"error {video}: {exc}")
            gc.collect()
    finally:
        pose_lm.close()
    if log_path and failures:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    return stats


def _extract_body_video_with_lm(
    video_path: str,
    out_path: Path,
    pose_lm: PoseLandmarker,
    ts_start_ms: int,
    frame_stride: int = 1,
    frame_size: int = 224,
) -> int:
    """Extract one clip; returns next timestamp for shared VIDEO-mode landmarker."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ts_start_ms
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seq: list[np.ndarray] = []
    ts = ts_start_ms
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (frame_i % frame_stride) != 0:
                frame_i += 1
                continue
            small = cv2.resize(frame, (frame_size, frame_size))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if seq or ts > ts_start_ms:
                ts += max(1, int(1000 / fps)) * frame_stride
            pr = pose_lm.detect_for_video(mp_img, ts)
            seq.append(body_from_pose(pr))
            del frame, small, rgb, mp_img, pr
            frame_i += 1
    finally:
        cap.release()
    if not seq:
        return ts_start_ms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.stack(seq, axis=0).astype(np.float32))
    return ts + 1


def main():
    ap = argparse.ArgumentParser(
        description="Extract MSPT body stream (memory-safe: one video at a time)."
    )
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    ap.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame to reduce memory/time on long clips.",
    )
    args = ap.parse_args()
    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    lab_root = C._resolve_lab_root()
    out_dir = lab_root / "cache" / "mspt_body"
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_model = C.INCLUDE_ML_ROOT / "models" / "pose_landmarker_full.task"
    if not pose_model.is_file():
        raise FileNotFoundError(f"Pose model missing: {pose_model}")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    totals = {"ok": 0, "skipped_existing": 0, "missing_video": 0, "failed": 0}
    log_dir = lab_root / "cache" / "mspt_body"
    for split in splits:
        mp = lab_root / "manifests" / f"{split}.csv"
        if mp.exists():
            st = process_manifest(
                mp, out_dir, pose_model, force=args.force,
                frame_stride=args.frame_stride, log_path=log_dir / f"failures_{split}.log",
            )
            for k in totals:
                totals[k] += st[k]
            print(f"  {split}: ok={st['ok']} skip={st['skipped_existing']} missing={st['missing_video']} failed={st['failed']}")
    n_body = sum(1 for _ in out_dir.rglob("*.npy"))
    n_lm = sum(1 for _ in (lab_root / "cache" / "landmarks").rglob("*.npy"))
    print(f"mspt body -> {out_dir}: {n_body}/{n_lm} files | totals {totals}")


if __name__ == "__main__":
    main()
