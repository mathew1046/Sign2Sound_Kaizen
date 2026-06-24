#!/usr/bin/env python3
"""Build train/val/test manifests for rtmlib MSPT lab (INCLUDE-50 + INCLUDE-263)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "data" / "include50_rtmlib_1080"
ORIG = REPO / "data" / "include50_rtmlib_1080" / "manifests"
VOCAB = REPO / "scripts" / "mspt" / "include50_mspt_and_include263_vocabulary.csv"
CACHE = LAB / "cache"


def _hash_split(word: str, stem: str) -> str:
    """Deterministic 85/7.5/7.5 split for clips without an original INCLUDE-50 split."""
    h = int(hashlib.md5(f"{word}/{stem}".encode()).hexdigest(), 16) % 100
    if h < 85:
        return "train"
    if h < 93:
        return "val"
    return "test"


def main() -> dict:
    man_path = LAB / "manifest.csv"
    if not man_path.is_file():
        raise FileNotFoundError(f"Missing {man_path} — download from Modal first")

    man = pd.read_csv(man_path)
    lid = {(str(r.label).strip(), str(r.stem).strip()): int(r.label_id) for r in man.itertuples()}

    orig_split: dict[tuple[str, str], str] = {}
    for split in ("train", "val", "test"):
        split_path = ORIG / f"{split}.csv"
        if not split_path.is_file():
            continue
        for _, row in pd.read_csv(split_path).iterrows():
            orig_split[(row["label"].strip(), Path(row["path"]).stem)] = split

    buckets: dict[str, list[dict]] = {s: [] for s in ("train", "val", "test")}
    unmatched = 0
    missing_id = 0

    for p in sorted((CACHE / "left_hand").rglob("*.npy")):
        label, stem = p.parent.name, p.stem
        key = (label, stem)
        split = orig_split.get(key)
        if split is None:
            unmatched += 1
            split = _hash_split(label, stem)
        label_id = lid.get(key)
        if label_id is None:
            missing_id += 1
            continue
        buckets[split].append(
            {
                "path": str(p),
                "label": label,
                "label_id": label_id,
                "stem": stem,
                "split": split,
            }
        )

    out_dir = LAB / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, rows in buckets.items():
        path = out_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "label_id", "split", "stem"])
            w.writeheader()
            w.writerows(rows)
        counts[split] = len(rows)

    n_classes = max((r["label_id"] for split_rows in buckets.values() for r in split_rows), default=-1) + 1
    summary = {
        "total_clips": sum(counts.values()),
        "splits": counts,
        "num_classes": n_classes,
        "unmatched_split": unmatched,
        "missing_label_id": missing_id,
    }
    (LAB / "lab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
