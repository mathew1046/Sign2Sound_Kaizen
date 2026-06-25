#!/usr/bin/env python3
"""Stitch isolated rtmlib clips into synthetic continuous signing sessions.

Produces per-frame sign/idle labels for training the learned segmenter.

Usage:
  python scripts/mspt/build_continuous_rtmlib_sessions.py \\
    --lab-root data/include50_rtmlib_1080 \\
    --num-sessions 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mspt.rtmlib_preprocess import (  # noqa: E402
    BODY_SLICE,
    FACE_SLICE,
    LEFT_HAND_SLICE,
    NUM_WHOLEBODY,
    RIGHT_HAND_SLICE,
)

FOOT_SLICE = slice(17, 23)


def streams_to_wholebody(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    body: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    """``(T, *, 4)`` per-stream caches -> ``(T, 133, 4)`` wholebody."""
    t = min(len(left_hand), len(right_hand), len(body), len(face))
    wb = np.zeros((t, NUM_WHOLEBODY, 4), dtype=np.float32)
    wb[:, LEFT_HAND_SLICE] = left_hand[:t]
    wb[:, RIGHT_HAND_SLICE] = right_hand[:t]
    wb[:, BODY_SLICE] = body[:t, :17]
    wb[:, FACE_SLICE] = face[:t, :68]
    return wb


def load_clip_wholebody(cache_root: Path, label: str, stem: str) -> np.ndarray:
    lh = np.load(cache_root / "left_hand" / label / f"{stem}.npy").astype(np.float32)
    rh = np.load(cache_root / "right_hand" / label / f"{stem}.npy").astype(np.float32)
    body = np.load(cache_root / "body" / label / f"{stem}.npy").astype(np.float32)
    face = np.load(cache_root / "face" / label / f"{stem}.npy").astype(np.float32)
    return streams_to_wholebody(lh, rh, body, face)


def build_sessions(
    manifest_csv: Path,
    cache_root: Path,
    out_dir: Path,
    num_sessions: int = 500,
    min_signs: int = 2,
    max_signs: int = 5,
    idle_min: int = 5,
    idle_max: int = 15,
    max_frames: int = 256,
    seed: int = 42,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    df = pd.read_csv(manifest_csv)
    by_label: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = row["stem"]
        label = row["label"]
        paths = [
            cache_root / "left_hand" / label / f"{stem}.npy",
            cache_root / "right_hand" / label / f"{stem}.npy",
            cache_root / "body" / label / f"{stem}.npy",
            cache_root / "face" / label / f"{stem}.npy",
        ]
        if all(p.is_file() for p in paths):
            by_label.setdefault(label, []).append(
                {"stem": stem, "label": label, "label_id": int(row["label_id"])}
            )

    labels = [k for k, v in by_label.items() if v]
    if not labels:
        print("[build] no clips with full cache found")
        return 0

    rows = []
    for i in range(num_sessions):
        sid = f"rtmlibsession_{i:05d}"
        n_signs = rng.randint(min_signs, min(max_signs, len(labels)))
        chosen = rng.sample(labels, n_signs)
        frames_acc: list[np.ndarray] = []
        label_acc: list[int] = []
        names: list[str] = []

        for lab in chosen:
            pick = rng.choice(by_label[lab])
            clip = load_clip_wholebody(cache_root, pick["label"], pick["stem"])
            total = sum(len(f) for f in frames_acc)
            if total + len(clip) > max_frames:
                clip = clip[: max_frames - total]
            frames_acc.append(clip)
            label_acc.extend([1] * len(clip))
            names.append(lab)

            idle = rng.randint(idle_min, idle_max)
            if frames_acc and sum(len(f) for f in frames_acc) + idle <= max_frames:
                frames_acc.append(np.zeros((idle, NUM_WHOLEBODY, 4), np.float32))
                label_acc.extend([0] * idle)
            if sum(len(f) for f in frames_acc) >= max_frames:
                break

        if not frames_acc:
            continue
        session = np.concatenate(frames_acc, axis=0)[:max_frames]
        labels_arr = np.array(label_acc[:max_frames], dtype=np.int64)
        np.save(out_dir / f"{sid}_wholebody.npy", session)
        np.save(out_dir / f"{sid}_label.npy", labels_arr)
        rows.append(
            {
                "session_id": sid,
                "num_frames": len(session),
                "glosses": "|".join(names),
                "split": "train",
            }
        )

    rng.shuffle(rows)
    n_val = max(1, int(0.15 * len(rows)))
    for j, r in enumerate(rows):
        r["split"] = "val" if j < n_val else "train"
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"[build] wrote {len(rows)} sessions to {out_dir}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build synthetic continuous rtmlib sessions")
    ap.add_argument("--lab-root", type=Path, default=REPO_ROOT / "data" / "include50_rtmlib_1080")
    ap.add_argument("--manifest", type=Path, default=None, help="Defaults to lab-root/manifests/train.csv")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-sessions", type=int, default=500)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lab = args.lab_root.resolve()
    manifest = args.manifest or (lab / "manifests" / "train.csv")
    out_dir = args.out_dir or (lab / "continuous")
    cache_root = lab / "cache"

    n = build_sessions(
        manifest_csv=manifest,
        cache_root=cache_root,
        out_dir=out_dir,
        num_sessions=args.num_sessions,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
