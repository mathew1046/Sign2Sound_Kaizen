#!/usr/bin/env python3
"""Demo: hand keypoints stitched onto cached body/face landmarks.

Pipeline (panel 3): [cansik/yolo-hand-detection](https://github.com/cansik/yolo-hand-detection)
bbox on 224px frame -> crop -> YOLO26 hand pose on each crop -> stitch.

Renders: [ RGB | MediaPipe cache skeleton | Cansik+YOLO26 stitched skeleton ]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

import slr_common as C  # noqa: E402
from hand_detect.cansik_yolo import CansikHandDetector, expand_crop  # noqa: E402
from mspt.pose_utils import SCALE_TARGET_SHOULDER, _apply_scale  # noqa: E402
from mspt.skeleton_viz import render_skeleton_panel  # noqa: E402

DEFAULT_CLIPS = [
    ("you_plural", "MVI_0022"),
    ("fan", "MVI_4533"),
    ("white", "MVI_5062"),
    ("loud", "MVI_5178"),
    ("cell_phone", "MVI_5393"),
]

FRAME_SIZE = 224  # match include50_lab preprocessing

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "yolo26" / "yolo26_hand_pose.pt"
DEFAULT_CANSIK_CFG = Path(__file__).resolve().parents[1] / "models" / "cansik" / "cross-hands-yolov4-tiny.cfg"
DEFAULT_CANSIK_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "cansik" / "cross-hands-yolov4-tiny.weights"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "collection_dashboard" / "evals" / "yolo26_demo"


def _valid_pt(xy: np.ndarray) -> bool:
    return bool((xy != 0).any() and 0 <= xy[0] <= 1 and 0 <= xy[1] <= 1)


def scale_transform_from_body(body_xy: np.ndarray):
    if body_xy.shape[0] <= 12:
        return None
    ls, rs = body_xy[11], body_xy[12]
    if not _valid_pt(ls) or not _valid_pt(rs):
        return None
    dist = float(np.linalg.norm(ls - rs))
    if dist <= 1e-6:
        return None
    cx, cy = (ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5
    return cx, cy, SCALE_TARGET_SHOULDER / dist


def hands_detected_frame(hands: np.ndarray) -> bool:
    for i in range(min(42, len(hands))):
        if _valid_pt(hands[i]):
            return True
    return False


def assign_hands_to_slots(
    candidates: list[tuple[float, str, np.ndarray]],
) -> np.ndarray:
    out = np.zeros((42, 2), dtype=np.float32)
    for side, base in (("left", 0), ("right", 21)):
        side_cands = [c for c in candidates if c[1] == side]
        if not side_cands:
            continue
        _, _, best = max(side_cands, key=lambda x: x[0])
        out[base : base + 21] = best[:21, :2]
    return out


def _hand_side(wrist: np.ndarray, body_xy: np.ndarray) -> str:
    body_lw = body_xy[15] if body_xy.shape[0] > 15 else np.zeros(2)
    body_rw = body_xy[16] if body_xy.shape[0] > 16 else np.zeros(2)
    if _valid_pt(body_lw) and _valid_pt(body_rw):
        d_left = float(np.linalg.norm(wrist - body_lw))
        d_right = float(np.linalg.norm(wrist - body_rw))
        return "left" if d_left <= d_right else "right"
    mid_x = 0.5
    if _valid_pt(body_xy[11]) and _valid_pt(body_xy[12]):
        mid_x = (body_xy[11, 0] + body_xy[12, 0]) * 0.5
    return "left" if wrist[0] <= mid_x else "right"


def yolo_hands_to_normalized(
    keypoints_xy: np.ndarray,
    confs: np.ndarray,
    frame_w: int,
    frame_h: int,
    body_xy: np.ndarray,
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """Map full-frame YOLO pose detections into ``(42, 2)`` hand array."""
    scale_tf = scale_transform_from_body(body_xy)

    candidates: list[tuple[float, str, np.ndarray]] = []
    for i in range(len(keypoints_xy)):
        if float(confs[i]) < conf_thresh:
            continue
        kpts = keypoints_xy[i].astype(np.float32).copy()
        kpts[:, 0] /= max(frame_w, 1)
        kpts[:, 1] /= max(frame_h, 1)
        for j in range(len(kpts)):
            kpts[j, 0], kpts[j, 1] = _apply_scale(float(kpts[j, 0]), float(kpts[j, 1]), scale_tf)

        wrist = kpts[0]
        side = _hand_side(wrist, body_xy)
        candidates.append((float(confs[i]), side, kpts))

    return assign_hands_to_slots(candidates)


def cansik_crop_yolo_hands(
    frame: np.ndarray,
    detector: CansikHandDetector,
    pose_model: YOLO,
    body_xy: np.ndarray,
    conf_thresh: float = 0.2,
    device: str | int = 0,
) -> np.ndarray:
    """Cansik bbox -> crop -> YOLO26 pose per crop -> normalized hand slots."""
    fh, fw = frame.shape[:2]
    scale_tf = scale_transform_from_body(body_xy)
    dets, _ = detector.detect(frame)
    candidates: list[tuple[float, str, np.ndarray]] = []

    for det_conf, x, y, w, h in dets:
        x0, y0, cw, ch = expand_crop(x, y, w, h, fw, fh)
        if cw < 12 or ch < 12:
            continue
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        results = pose_model.predict(crop, imgsz=320, conf=conf_thresh, verbose=False, device=device)
        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        best_i = int(r.boxes.conf.argmax())
        pose_conf = float(r.boxes.conf[best_i])
        kpts = r.keypoints.xy[best_i].cpu().numpy().astype(np.float32)
        full = np.zeros((21, 2), dtype=np.float32)
        full[:, 0] = (kpts[:, 0] + x0) / max(fw, 1)
        full[:, 1] = (kpts[:, 1] + y0) / max(fh, 1)
        for j in range(21):
            full[j, 0], full[j, 1] = _apply_scale(float(full[j, 0]), float(full[j, 1]), scale_tf)
        side = _hand_side(full[0], body_xy)
        candidates.append((det_conf * pose_conf, side, full))

    return assign_hands_to_slots(candidates)


@dataclass
class ClipPaths:
    word: str
    stem: str
    video: Path
    landmarks: Path
    body: Path
    face: Path


def resolve_clip(lab_root: Path, word: str, stem: str) -> ClipPaths | None:
    manifest_dir = lab_root / "manifests"
    video: Path | None = None
    for csv_path in sorted(manifest_dir.glob("*.csv")):
        if csv_path.name == "all.csv":
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"].strip() != word:
                    continue
                p = C.resolve_video_path(row["path"].strip())
                if p.stem == stem and p.is_file():
                    video = p
                    break
        if video is not None:
            break
    if video is None:
        return None
    return ClipPaths(
        word=word,
        stem=stem,
        video=video,
        landmarks=lab_root / "cache" / "landmarks" / word / f"{stem}.npy",
        body=lab_root / "cache" / "mspt_body" / word / f"{stem}.npy",
        face=lab_root / "cache" / "landmarks_face" / word / f"{stem}.npy",
    )


def load_cached_sequences(paths: ClipPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lm = np.load(paths.landmarks, mmap_mode="r")
    mp_hands = np.array(lm[:, 12:54, :2], dtype=np.float32)
    t = len(mp_hands)
    if paths.body.is_file():
        body = np.array(np.load(paths.body, mmap_mode="r")[:, :33, :2], dtype=np.float32)
    else:
        body = np.zeros((t, 33, 2), dtype=np.float32)
        body[:, :12] = np.array(lm[:, :12, :2], dtype=np.float32)
    if paths.face.is_file():
        face = np.array(np.load(paths.face, mmap_mode="r")[..., :2], dtype=np.float32)
    else:
        face = np.zeros((t, 72, 2), dtype=np.float32)
    del lm
    return mp_hands, body, face


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 24), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1, cv2.LINE_AA)
    return out


def process_clip(
    clip: ClipPaths,
    pose_model: YOLO,
    detector: CansikHandDetector,
    out_dir: Path,
    panel_size: int = 400,
    conf: float = 0.25,
    device: str | int = 0,
) -> dict:
    mp_hands, body, face = load_cached_sequences(clip)
    n_frames = len(mp_hands)

    cap = cv2.VideoCapture(str(clip.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {clip.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip.word}_{clip.stem}.mp4"
    row_h = panel_size
    out_w = panel_size * 3
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, row_h),
    )

    mp_detect = 0
    stitch_detect = 0
    frame_i = 0

    while frame_i < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        mp_h = mp_hands[frame_i]
        b = body[frame_i]
        f = face[frame_i]

        small = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        stitch_h = cansik_crop_yolo_hands(
            small, detector, pose_model, b, conf_thresh=conf, device=device,
        )

        if hands_detected_frame(mp_h):
            mp_detect += 1
        if hands_detected_frame(stitch_h):
            stitch_detect += 1

        rgb_panel = cv2.resize(small, (panel_size, panel_size))
        mp_panel = label_panel(render_skeleton_panel(mp_h, b, f, panel_size=panel_size), "MediaPipe cache")
        stitch_panel = label_panel(
            render_skeleton_panel(stitch_h, b, f, panel_size=panel_size),
            "Cansik bbox + YOLO26",
        )
        row = np.hstack([rgb_panel, mp_panel, stitch_panel])
        writer.write(row)
        frame_i += 1

    cap.release()
    writer.release()

    used = max(frame_i, 1)
    stats = {
        "word": clip.word,
        "stem": clip.stem,
        "video": str(clip.video),
        "output": str(out_path),
        "frames": frame_i,
        "fps": float(fps),
        "mp_hand_frames": mp_detect,
        "stitch_hand_frames": stitch_detect,
        "mp_hand_rate": round(mp_detect / used, 4),
        "stitch_hand_rate": round(stitch_detect / used, 4),
        "pipeline": "cansik_yolov4_tiny_bbox -> yolo26_hand_pose_crop",
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO26 hand stitch demo on INCLUDE-50 clips")
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cansik-cfg", type=Path, default=DEFAULT_CANSIK_CFG)
    ap.add_argument("--cansik-weights", type=Path, default=DEFAULT_CANSIK_WEIGHTS)
    ap.add_argument("--cansik-size", type=int, default=256)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--panel-size", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)
    ap.add_argument(
        "--clips",
        nargs="*",
        default=[f"{w}/{s}" for w, s in DEFAULT_CLIPS],
        help="word/stem pairs, e.g. fan/MVI_4533",
    )
    args = ap.parse_args()

    lab_root = Path(args.lab_root) if args.lab_root else C._resolve_lab_root()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.cansik_cfg.is_file() or not args.cansik_weights.is_file():
        raise FileNotFoundError(
            f"Cansik model missing: {args.cansik_cfg} / {args.cansik_weights}"
        )

    pose_model = YOLO(str(args.model))
    detector = CansikHandDetector(
        args.cansik_cfg, args.cansik_weights,
        size=args.cansik_size, confidence=args.conf,
    )
    all_stats: list[dict] = []

    for spec in args.clips:
        word, stem = spec.split("/", 1)
        clip = resolve_clip(lab_root, word, stem)
        if clip is None:
            print(f"[skip] could not resolve {word}/{stem}")
            continue
        if not clip.landmarks.is_file():
            print(f"[skip] no landmark cache for {word}/{stem}")
            continue
        print(f"[run] {word}/{stem} ...")
        stats = process_clip(
            clip, pose_model, detector, args.out_dir,
            panel_size=args.panel_size, conf=args.conf, device=args.device,
        )
        all_stats.append(stats)
        print(
            f"  -> {stats['output']} | MP hands {stats['mp_hand_rate']:.1%} | "
            f"Cansik+YOLO {stats['stitch_hand_rate']:.1%}"
        )

    summary_path = args.out_dir / "summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\n[done] {len(all_stats)} videos -> {args.out_dir}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
