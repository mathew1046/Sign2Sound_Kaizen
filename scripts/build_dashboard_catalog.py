#!/usr/bin/env python3
"""Build dashboard catalog: score rtmlib wholebody clips and pick default exemplar per gloss."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config import (  # noqa: E402
    CATALOG_V263,
    ensure_combined_manifest,
    MANIFEST_PATH,
    WHOLEBODY_DIR,
)
from dashboard.label_utils import ensure_label_map, load_label_map, signer_id_from_path  # noqa: E402


def wholebody_path(label: str, stem: str, root: Path) -> Path:
    return root / label / f"{stem}.npy"


def visibility_score(wb_path: Path, sample_every: int = 5) -> float:
    """Mean fraction of visible wholebody keypoints on sampled frames."""
    if not wb_path.exists():
        return 0.0
    seq = np.load(wb_path, mmap_mode="r")
    if seq.shape[0] == 0:
        return 0.0
    idxs = range(0, seq.shape[0], max(1, sample_every))
    scores = []
    for t in idxs:
        frame = np.asarray(seq[t])
        if frame.shape[-1] >= 4:
            vis = frame[..., 3] > 0
            scores.append(float(np.mean(vis)))
        else:
            scores.append(float(np.mean(frame[..., :2] > 0.01)))
    return float(np.mean(scores)) if scores else 0.0


def frame_signature(wb_path: Path, n_bins: int = 16) -> np.ndarray:
    seq = np.load(wb_path, mmap_mode="r")
    sigs = []
    step = max(1, seq.shape[0] // 12)
    for t in range(0, seq.shape[0], step):
        frame = np.asarray(seq[t])
        xy = frame[..., :2].astype(np.float32)
        flat = xy.reshape(-1)
        hist, _ = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
        sigs.append(hist.astype(np.float32))
    if not sigs:
        return np.zeros((1, n_bins), dtype=np.float32)
    return np.stack(sigs, axis=0)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.shape[0], b.shape[0]
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb])


def pick_medoid(candidates: list[dict], root: Path) -> dict:
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 8:
        return max(candidates, key=lambda c: c["visibility_score"])

    def _path(c: dict) -> Path:
        rel = Path(c["skeleton_path"])
        if rel.is_file():
            return rel
        if (ROOT / rel).is_file():
            return ROOT / rel
        return root / c["label"] / f"{c['exemplar_id']}.npy"

    paths = [_path(c) for c in candidates]
    sigs = [frame_signature(p) for p in paths]
    best_idx = 0
    best_sum = float("inf")
    for i, si in enumerate(sigs):
        total = sum(dtw_distance(si, sj) for j, sj in enumerate(sigs) if i != j)
        if total < best_sum or (
            abs(total - best_sum) < 1e-6
            and candidates[i]["visibility_score"] > candidates[best_idx]["visibility_score"]
        ):
            best_sum = total
            best_idx = i
    return candidates[best_idx]


def build_catalog(
    manifest_path: Path,
    wholebody_root: Path,
    label_map_path: Path,
    out_path: Path,
    vocab_version: int = 263,
) -> dict:
    ensure_label_map(label_map_path)
    label_map, _ = load_label_map(str(label_map_path))

    by_label: dict[str, list[dict]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            stem = row.get("stem") or Path(row["path"]).stem
            wb_path = wholebody_path(label, stem, wholebody_root)
            if not wb_path.exists():
                continue
            seq = np.load(wb_path, mmap_mode="r")
            num_frames = int(seq.shape[0])
            vis = visibility_score(wb_path)
            signer = signer_id_from_path(row["path"])
            rel = (
                str(wb_path.relative_to(ROOT))
                if wb_path.is_relative_to(ROOT)
                else f"{label}/{stem}.npy"
            )
            by_label.setdefault(label, []).append(
                {
                    "exemplar_id": stem,
                    "signer_id": signer,
                    "label": label,
                    "label_id": int(row["label_id"]),
                    "num_frames": num_frames,
                    "visibility_score": round(vis, 4),
                    "skeleton_path": rel,
                    "split": row.get("split", ""),
                }
            )

    glosses = []
    for gloss, label_id in sorted(label_map.items(), key=lambda x: x[1]):
        variants = by_label.get(gloss, [])
        entry = {
            "gloss": gloss,
            "label_id": label_id,
            "display_name": gloss.replace("_", " ").title(),
            "variant_count": len(variants),
            "variants": sorted(variants, key=lambda v: -v["visibility_score"]),
            "default_exemplar_id": None,
        }
        if variants:
            default = pick_medoid(variants, wholebody_root)
            entry["default_exemplar_id"] = default["exemplar_id"]
            entry["default_num_frames"] = default["num_frames"]
        glosses.append(entry)

    catalog = {
        "vocab_version": vocab_version,
        "project_root": str(ROOT),
        "skeleton_dir": str(
            wholebody_root.relative_to(ROOT)
            if wholebody_root.is_relative_to(ROOT)
            else wholebody_root
        ),
        "num_glosses": len(label_map),
        "glosses_with_data": sum(1 for g in glosses if g["default_exemplar_id"]),
        "glosses": glosses,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--skeleton-dir", type=Path, default=WHOLEBODY_DIR)
    parser.add_argument("--label-map", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=CATALOG_V263)
    parser.add_argument("--vocab-version", type=int, default=263)
    args = parser.parse_args()

    manifest = args.manifest or ensure_combined_manifest()
    label_map = args.label_map or (ROOT / "dashboard" / "label_map.json")

    catalog = build_catalog(
        manifest,
        args.skeleton_dir,
        label_map,
        args.out,
        vocab_version=args.vocab_version,
    )
    print(
        f"Wrote {args.out}: {catalog['glosses_with_data']}/{catalog['num_glosses']} glosses with exemplars"
    )


if __name__ == "__main__":
    main()
