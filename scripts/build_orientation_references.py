#!/usr/bin/env python3
"""Build orientation reference JSON files from catalog_v263 + wholebody RTMLIB caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config import CATALOG_V263, ORIENTATION_REFS_DIR, PROJECT_ROOT  # noqa: E402
from dashboard.server.orientation.features import (  # noqa: E402
    extract_sequence_features,
    palm_normal_angle_deg,
    sequence_from_wholebody,
)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    c = PROJECT_ROOT / path_str
    if c.is_file():
        return c
    return p


def _temporal_variance(ref_seq: list[dict]) -> float:
    if len(ref_seq) < 2:
        return 0.0
    vals = []
    for i in range(1, len(ref_seq)):
        vals.append(
            palm_normal_angle_deg(
                ref_seq[i - 1].get("palm_normal", [0, 0, 0]),
                ref_seq[i].get("palm_normal", [0, 0, 0]),
            )
        )
    return float(np.mean(vals)) if vals else 0.0


def _variant_mean_features(seq: list[dict]) -> dict:
    from dashboard.server.orientation.compare import _average_features

    return _average_features(seq)


def _compute_tolerances(variant_means: list[dict], floors: dict[str, float], k: float = 2.0) -> dict[str, float]:
    palm_devs = []
    flex_devs = []
    curl_devs = []

    if len(variant_means) < 2:
        return {
            "palm_normal_deg": floors["palm_normal_deg"],
            "wrist_flexion_deg": floors["wrist_flexion_deg"],
            "finger_curl_deg": floors["finger_curl_deg"],
        }

    ref = variant_means[0]
    for vm in variant_means[1:]:
        palm_devs.append(
            palm_normal_angle_deg(ref.get("palm_normal", [0, 0, 0]), vm.get("palm_normal", [0, 0, 0]))
        )
        flex_devs.append(abs(float(ref.get("wrist_flexion_deg", 0)) - float(vm.get("wrist_flexion_deg", 0))))
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            rc = ref.get("finger_curls", {}).get(finger, [])
            vc = vm.get("finger_curls", {}).get(finger, [])
            for a, b in zip(rc, vc):
                curl_devs.append(abs(float(a) - float(b)))

    return {
        "palm_normal_deg": max(floors["palm_normal_deg"], k * float(np.std(palm_devs)) if palm_devs else 0),
        "wrist_flexion_deg": max(floors["wrist_flexion_deg"], k * float(np.std(flex_devs)) if flex_devs else 0),
        "finger_curl_deg": max(floors["finger_curl_deg"], k * float(np.std(curl_devs)) if curl_devs else 0),
    }


def build_gloss_reference(gloss_entry: dict, fps: float = 25.0) -> dict | None:
    gloss = gloss_entry["gloss"]
    default_id = gloss_entry.get("default_exemplar_id")
    variants = gloss_entry.get("variants", [])
    if not variants:
        return None

    variant_seqs: list[tuple[str, list[dict]]] = []
    for v in variants:
        skel = _resolve_path(v["skeleton_path"])
        if not skel.is_file():
            continue
        wb = np.load(skel, mmap_mode="r")
        feats, hand = sequence_from_wholebody(np.asarray(wb), fps=fps)
        variant_seqs.append((v["exemplar_id"], feats))

    if not variant_seqs:
        return None

    ref_pair = next((p for p in variant_seqs if p[0] == default_id), variant_seqs[0])
    ref_seq = ref_pair[1]
    active_hand = ref_seq[0].get("active_hand", "right") if ref_seq else "right"

    variant_means = [_variant_mean_features(seq) for _, seq in variant_seqs]
    floors = {"palm_normal_deg": 12.0, "wrist_flexion_deg": 8.0, "finger_curl_deg": 10.0}
    tolerance = _compute_tolerances(variant_means, floors)

    motion = _temporal_variance(ref_seq)
    sign_type = "static" if motion < 8.0 else "dynamic"

    critical = ["palm_normal", "wrist_flexion_deg", "finger_curls"]
    if sign_type == "static":
        critical = ["palm_normal", "wrist_flexion_deg", "finger_curls"]

    return {
        "sign_id": gloss,
        "display_name": gloss_entry.get("display_name", gloss.replace("_", " ").title()),
        "sign_type": sign_type,
        "active_hand": active_hand,
        "critical_features": critical,
        "reference_sequence": ref_seq,
        "tolerance": tolerance,
        "default_exemplar_id": ref_pair[0],
        "variant_count": len(variant_seqs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build orientation reference library")
    parser.add_argument("--catalog", type=Path, default=CATALOG_V263)
    parser.add_argument("--out", type=Path, default=ORIENTATION_REFS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Process only N glosses (0 = all)")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    glosses = catalog.get("glosses", [])
    if args.limit > 0:
        glosses = glosses[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    built = 0
    skipped = 0

    for entry in tqdm(glosses, desc="orientation refs"):
        ref = build_gloss_reference(entry)
        if ref is None:
            skipped += 1
            continue
        gloss = ref["sign_id"]
        out_name = f"{gloss}.json"
        out_path = args.out / out_name
        out_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        index[gloss] = out_name
        built += 1

    index_doc = {
        "vocab_version": catalog.get("vocab_version", 263),
        "num_glosses": built,
        "skipped": skipped,
        "glosses": index,
    }
    (args.out / "index.json").write_text(json.dumps(index_doc, indent=2), encoding="utf-8")
    print(f"Built {built} references ({skipped} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
