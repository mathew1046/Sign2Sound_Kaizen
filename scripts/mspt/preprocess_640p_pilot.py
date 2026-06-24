#!/usr/bin/env python3
"""Pilot: preprocess INCLUDE-50 clips at 640×640, source FPS, MediaPipe GPU.

Extracts landmarks + body + face, compares hand visibility vs existing 224 cache,
and writes side-by-side preview videos.

Usage:
  export INCLUDE50_LAB_ROOT=/media/mathew/OS/Users/augus/INCLUDE_ML/include50_lab
  export INCLUDE_ML_ROOT=/media/mathew/OS/Users/augus/INCLUDE_ML
  cd notebooks && python preprocess_640p_pilot.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

import slr_common as C  # noqa: E402
from mspt.live_extract import face_from_result  # noqa: E402
from mspt.pose_utils import NUM_POSE, body_from_pose, landmarks_from_results  # noqa: E402
from mspt.skeleton_viz import render_skeleton_panel  # noqa: E402

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FRAME_SIZE = 640
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "collection_dashboard"
    / "evals"
    / "preprocess_640p_pilot"
)

# Mix of high/low hand-visibility clips from prior analysis
DEFAULT_CLIPS = [
    ("you_plural", "MVI_0022"),
    ("bird", "MVI_3012"),
    ("fan", "MVI_4533"),
    ("loud", "MVI_5178"),
    ("thank_you", "MVI_0006"),
    ("hot", "MVI_9249"),
    ("white", "MVI_5062"),
    ("paint", "MVI_4415"),
    ("teacher", "MVI_4458"),
    ("cell_phone", "MVI_5393"),
]


def _valid_pt(xy: np.ndarray) -> bool:
    return bool((xy != 0).any() and 0 <= xy[0] <= 1 and 0 <= xy[1] <= 1)


def hand_stats(hands: np.ndarray) -> dict[str, int | float]:
    lh = hands[:21]
    rh = hands[21:42]
    left = sum(1 for p in lh if _valid_pt(p))
    right = sum(1 for p in rh if _valid_pt(p))
    return {
        "left_joints": left,
        "right_joints": right,
        "left_hand": int(left > 0),
        "right_hand": int(right > 0),
        "any_hand": int(left > 0 or right > 0),
    }


def sequence_hand_rates(hands_seq: np.ndarray) -> dict[str, float]:
    n = max(len(hands_seq), 1)
    left = right = any_h = 0
    for h in hands_seq:
        st = hand_stats(h)
        left += st["left_hand"]
        right += st["right_hand"]
        any_h += st["any_hand"]
    return {
        "left_hand_rate": round(left / n, 4),
        "right_hand_rate": round(right / n, 4),
        "any_hand_rate": round(any_h / n, 4),
        "frames": len(hands_seq),
    }


@dataclass
class ClipRef:
    word: str
    stem: str
    video: Path


def resolve_clip(lab_root: Path, word: str, stem: str) -> ClipRef | None:
    for csv_path in sorted((lab_root / "manifests").glob("*.csv")):
        if csv_path.name == "all.csv":
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"].strip() != word:
                    continue
                p = C.resolve_video_path(row["path"].strip())
                if p.stem == stem and p.is_file():
                    return ClipRef(word=word, stem=stem, video=p)
    return None


def create_gpu_landmarkers(models_dir: Path):
    """MediaPipe Tasks with GPU delegate (see Google Edge MP docs)."""
    gpu = BaseOptions.Delegate.GPU
    pose_path = models_dir / "pose_landmarker_full.task"
    hand_path = models_dir / "hand_landmarker.task"
    face_path = models_dir / "face_landmarker.task"
    for p in (pose_path, hand_path, face_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing model: {p}")

    pose_lm = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path), delegate=gpu),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
        )
    )
    hand_lm = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path), delegate=gpu),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_lm = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_path), delegate=gpu),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
        )
    )
    return pose_lm, hand_lm, face_lm


def extract_clip_gpu(
    video_path: Path,
    pose_lm: PoseLandmarker,
    hand_lm: HandLandmarker,
    face_lm: FaceLandmarker,
    ts_start_ms: int = 0,
    frame_size: int = FRAME_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Stream video at source FPS; return landmarks, body, face, fps, next_ts_ms."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    lm_seq: list[np.ndarray] = []
    body_seq: list[np.ndarray] = []
    face_seq: list[np.ndarray] = []
    ts = ts_start_ms

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (frame_size, frame_size))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if lm_seq or ts > ts_start_ms:
                ts += max(1, int(1000 / fps))
            pr = pose_lm.detect_for_video(mp_img, ts)
            hr = hand_lm.detect_for_video(mp_img, ts)
            fr = face_lm.detect_for_video(mp_img, ts)
            lm_seq.append(landmarks_from_results(pr, hr))
            body_seq.append(body_from_pose(pr))
            face_seq.append(face_from_result(fr))
    finally:
        cap.release()

    if not lm_seq:
        raise RuntimeError(f"No frames decoded from {video_path}")

    lm = np.stack(lm_seq, axis=0).astype(np.float32)
    body = np.stack(body_seq, axis=0).astype(np.float32)
    face = np.stack(face_seq, axis=0).astype(np.float32)
    return lm, body, face, fps, ts + 1


def load_224_cache(
    lab_root: Path, word: str, stem: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    lm_p = lab_root / "cache" / "landmarks" / word / f"{stem}.npy"
    body_p = lab_root / "cache" / "mspt_body" / word / f"{stem}.npy"
    face_p = lab_root / "cache" / "landmarks_face" / word / f"{stem}.npy"
    if not lm_p.is_file():
        return None, None, None
    lm = np.load(lm_p, mmap_mode="r")
    hands = np.array(lm[:, 12:54, :2], dtype=np.float32)
    if body_p.is_file():
        body = np.array(np.load(body_p, mmap_mode="r")[:, :33, :2], dtype=np.float32)
    else:
        body = np.zeros((len(hands), 33, 2), dtype=np.float32)
        body[:, :12] = np.array(lm[:, :12, :2], dtype=np.float32)
    if face_p.is_file():
        face = np.array(np.load(face_p, mmap_mode="r")[..., :2], dtype=np.float32)
    else:
        face = np.zeros((len(hands), 72, 2), dtype=np.float32)
    return hands, body, face


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (18, 18, 22), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 240), 1, cv2.LINE_AA)
    return out


def render_preview(
    video_path: Path,
    hands_640: np.ndarray,
    body_640: np.ndarray,
    face_640: np.ndarray,
    hands_224: np.ndarray | None,
    body_224: np.ndarray | None,
    face_224: np.ndarray | None,
    out_path: Path,
    fps: float,
    stats_640: dict,
    stats_224: dict | None,
    panel_size: int = 400,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    n = len(hands_640)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_size * 3, panel_size),
    )

    title_224 = "224 cache"
    if stats_224:
        title_224 += f" hands {stats_224['any_hand_rate']:.0%}"
    title_640 = f"640 GPU hands {stats_640['any_hand_rate']:.0%}"

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        rgb_panel = cv2.resize(small, (panel_size, panel_size))

        if hands_224 is not None and body_224 is not None and i < len(hands_224):
            skel_224 = render_skeleton_panel(
                hands_224[i],
                body_224[i],
                face_224[i] if face_224 is not None else None,
                panel_size=panel_size,
            )
        else:
            skel_224 = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            skel_224[:] = (28, 28, 32)
        skel_640 = render_skeleton_panel(
            hands_640[i], body_640[i, :, :2], face_640[i, :, :2], panel_size=panel_size,
        )
        row = np.hstack([
            label_panel(rgb_panel, f"RGB {FRAME_SIZE}px"),
            label_panel(skel_224, title_224),
            label_panel(skel_640, title_640),
        ])
        writer.write(row)

    cap.release()
    writer.release()


def process_clip(
    clip: ClipRef,
    lab_root: Path,
    cache_root: Path,
    preview_dir: Path,
    pose_lm,
    hand_lm,
    face_lm,
    ts_start_ms: int,
) -> tuple[dict, int]:
    t0 = time.time()
    lm, body, face, fps, ts_next = extract_clip_gpu(
        clip.video, pose_lm, hand_lm, face_lm, ts_start_ms=ts_start_ms,
    )
    elapsed = time.time() - t0

    hands_640 = lm[:, NUM_POSE : NUM_POSE + 42, :2]
    stats_640 = sequence_hand_rates(hands_640)

    lm_out = cache_root / "landmarks" / clip.word / f"{clip.stem}.npy"
    body_out = cache_root / "mspt_body" / clip.word / f"{clip.stem}.npy"
    face_out = cache_root / "landmarks_face" / clip.word / f"{clip.stem}.npy"
    for path, arr in ((lm_out, lm), (body_out, body), (face_out, face)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    hands_224, body_224, face_224 = load_224_cache(lab_root, clip.word, clip.stem)
    stats_224 = sequence_hand_rates(hands_224) if hands_224 is not None else None

    preview_path = preview_dir / f"{clip.word}_{clip.stem}.mp4"
    render_preview(
        clip.video, hands_640, body, face, hands_224, body_224, face_224,
        preview_path, fps, stats_640, stats_224,
    )

    return {
        "word": clip.word,
        "stem": clip.stem,
        "video": str(clip.video),
        "frames": int(len(lm)),
        "fps": fps,
        "frame_size": FRAME_SIZE,
        "delegate": "GPU",
        "extract_sec": round(elapsed, 2),
        "fps_extract": round(len(lm) / max(elapsed, 0.01), 1),
        "cache": {
            "landmarks": str(lm_out),
            "body": str(body_out),
            "face": str(face_out),
        },
        "preview": str(preview_path),
        "hand_stats_640": stats_640,
        "hand_stats_224": stats_224,
        "hand_rate_delta": (
            round(stats_640["any_hand_rate"] - stats_224["any_hand_rate"], 4)
            if stats_224 else None
        ),
    }, ts_next


def main() -> None:
    ap = argparse.ArgumentParser(description="640px GPU preprocessing pilot (10 clips)")
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--clips", nargs="*", default=[f"{w}/{s}" for w, s in DEFAULT_CLIPS],
    )
    args = ap.parse_args()

    lab_root = Path(args.lab_root) if args.lab_root else C._resolve_lab_root()
    cache_root = args.out_dir / "cache"
    preview_dir = args.out_dir / "previews"
    models_dir = C.INCLUDE_ML_ROOT / "models"

    print(f"[pilot] frame_size={FRAME_SIZE} delegate=GPU models={models_dir}")
    pose_lm, hand_lm, face_lm = create_gpu_landmarkers(models_dir)

    results: list[dict] = []
    ts_cursor = 0
    try:
        for spec in args.clips[: args.limit]:
            word, stem = spec.split("/", 1)
            clip = resolve_clip(lab_root, word, stem)
            if clip is None:
                print(f"[skip] missing clip {word}/{stem}")
                continue
            print(f"[run] {word}/{stem} ...")
            try:
                stats, ts_cursor = process_clip(
                    clip, lab_root, cache_root, preview_dir,
                    pose_lm, hand_lm, face_lm, ts_cursor,
                )
                results.append(stats)
                s640 = stats["hand_stats_640"]["any_hand_rate"]
                s224 = stats["hand_stats_224"]["any_hand_rate"] if stats["hand_stats_224"] else 0
                print(
                    f"  frames={stats['frames']} fps={stats['fps']:.1f} "
                    f"extract={stats['fps_extract']:.1f} fps | "
                    f"hands 224={s224:.0%} -> 640={s640:.0%} "
                    f"(delta {stats['hand_rate_delta']:+.0%})"
                )
            except Exception as exc:
                print(f"  [error] {exc}")
            gc.collect()
    finally:
        pose_lm.close()
        hand_lm.close()
        face_lm.close()

    summary = {
        "frame_size": FRAME_SIZE,
        "delegate": "GPU",
        "n_clips": len(results),
        "clips": results,
        "aggregate": {
            "mean_hand_rate_640": round(
                sum(r["hand_stats_640"]["any_hand_rate"] for r in results) / max(len(results), 1), 4
            ),
            "mean_hand_rate_224": round(
                sum(r["hand_stats_224"]["any_hand_rate"] for r in results if r["hand_stats_224"])
                / max(sum(1 for r in results if r["hand_stats_224"]), 1),
                4,
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {len(results)} clips")
    print(f"[done] cache -> {cache_root}")
    print(f"[done] previews -> {preview_dir}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
