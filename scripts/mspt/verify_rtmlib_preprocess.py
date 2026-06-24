#!/usr/bin/env python3
"""Verify rtmlib INCLUDE-50 preprocess: frame counts, shapes, split coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "data" / "include50_rtmlib_1080"
CACHE = LAB / "cache"
STREAMS = ("wholebody", "body", "foot", "face", "left_hand", "right_hand")


def main() -> int:
    errors: list[str] = []
    per_clip_list = json.loads((LAB / "per_clip.json").read_text(encoding="utf-8"))
    per_clip = {(e["word"], e["stem"]): e for e in per_clip_list if e.get("ok")}
    metrics = json.loads((LAB / "metrics.json").read_text(encoding="utf-8"))

    clips = sorted((CACHE / "left_hand").rglob("*.npy"))
    print(f"clips (left_hand): {len(clips)}")

    total_frames = 0
    shape_ok = 0
    for p in clips:
        word, stem = p.parent.name, p.stem
        paths = {s: CACHE / s / word / f"{stem}.npy" for s in STREAMS if s != "wholebody"}
        paths["wholebody"] = CACHE / "wholebody" / word / f"{stem}.npy"
        missing = [s for s, fp in paths.items() if not fp.is_file()]
        if missing:
            errors.append(f"{word}/{stem}: missing {missing}")
            continue

        arrays = {s: np.load(paths[s], mmap_mode="r") for s in paths}
        ts = {s: arr.shape[0] for s, arr in arrays.items()}
        if len(set(ts.values())) != 1:
            errors.append(f"{word}/{stem}: frame mismatch {ts}")
            continue

        t = ts["wholebody"]
        total_frames += t

        exp = per_clip.get((word, stem))
        if exp is not None and int(exp["frames"]) != t:
            errors.append(f"{word}/{stem}: per_clip frames {exp['frames']} != npy {t}")

        wb, body, face, lh, rh = (
            arrays["wholebody"], arrays["body"], arrays["face"],
            arrays["left_hand"], arrays["right_hand"],
        )
        ok = (
            wb.shape[1:] == (133, 4)
            and body.shape[1:] == (17, 4)
            and face.shape[1:] == (68, 4)
            and lh.shape[1:] == (21, 4)
            and rh.shape[1:] == (21, 4)
            and t > 0
        )
        if ok:
            shape_ok += 1
        else:
            errors.append(f"{word}/{stem}: bad shapes wb={wb.shape} body={body.shape}")

    print(f"shape_ok: {shape_ok}/{len(clips)}")
    print(f"total_frames (wholebody sum): {total_frames}")
    print(f"metrics.json total_frames: {metrics.get('total_frames')}")
    print(f"metrics clips_ok: {metrics.get('n_clips_ok')} failed={metrics.get('n_clips_failed')}")

    if total_frames != metrics.get("total_frames"):
        errors.append(f"frame sum mismatch: {total_frames} vs metrics {metrics.get('total_frames')}")

    man_path = LAB / "manifests" / "train.csv"
    if man_path.is_file():
        import pandas as pd
        n = sum(len(pd.read_csv(LAB / "manifests" / f"{s}.csv")) for s in ("train", "val", "test"))
        print(f"manifest clips: {n}")
        if n != len(clips):
            errors.append(f"manifest count {n} != cache {len(clips)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:30]:
            print(" ", e)
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 1

    print("\nOK: all clips verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
