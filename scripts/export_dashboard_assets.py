#!/usr/bin/env python3
"""Export rtmlib wholebody clips to PNG frames for fast browser loading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config import ASSETS_DIR, CATALOG_V263, WHOLEBODY_DIR  # noqa: E402
from mspt.rtmlib_skeleton_viz import render_rtmlib_skeleton_panel  # noqa: E402


def export_exemplar(wb_path: Path, out_dir: Path, panel_size: int = 480, png_compression: int = 3) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = np.load(wb_path, mmap_mode="r")
    n = int(seq.shape[0])
    for t in range(n):
        out = out_dir / f"{t:05d}.png"
        if out.exists():
            continue
        panel = render_rtmlib_skeleton_panel(seq[t], panel_size=panel_size)
        cv2.imwrite(str(out), panel, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_V263)
    parser.add_argument("--skeleton-dir", type=Path, default=WHOLEBODY_DIR)
    parser.add_argument("--out", type=Path, default=ASSETS_DIR)
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    tasks: list[tuple[str, str, Path]] = []

    for g in catalog["glosses"]:
        gloss = g["gloss"]
        if args.all_variants:
            for v in g.get("variants", []):
                sk = args.skeleton_dir / gloss / f"{v['exemplar_id']}.npy"
                tasks.append((gloss, v["exemplar_id"], sk))
        elif g.get("default_exemplar_id"):
            ex = g["default_exemplar_id"]
            sk = args.skeleton_dir / gloss / f"{ex}.npy"
            tasks.append((gloss, ex, sk))

    manifest_rows = []
    for gloss, ex_id, sk_path in tqdm(tasks, desc="export"):
        if not sk_path.exists():
            tqdm.write(f"skip missing {sk_path}")
            continue
        out_dir = args.out / gloss / ex_id
        n = export_exemplar(sk_path, out_dir)
        manifest_rows.append(
            {"gloss": gloss, "exemplar_id": ex_id, "num_frames": n, "asset_dir": str(out_dir.relative_to(ROOT))}
        )

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps({"exports": manifest_rows}, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest_rows)} exemplars -> {args.out}")


if __name__ == "__main__":
    main()
