#!/usr/bin/env python3
"""Extract landmark sidecars for catalog default exemplars (from source videos)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config import CATALOG_V50, LANDMARKS_DIR, MANIFEST_PATH  # noqa: E402
from include50_lab.preprocess.landmarks import extract_landmarks_video  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_V50)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=LANDMARKS_DIR)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    df = pd.read_csv(args.manifest)
    path_by_stem = {Path(r["path"]).stem: r["path"] for _, r in df.iterrows()}

    for g in tqdm(catalog["glosses"], desc="landmarks"):
        ex = g.get("default_exemplar_id")
        if not ex:
            continue
        out = args.out / g["gloss"] / f"{ex}.npy"
        if out.exists():
            continue
        video = path_by_stem.get(ex)
        if not video or not Path(video).exists():
            continue
        extract_landmarks_video(video, out)


if __name__ == "__main__":
    main()
