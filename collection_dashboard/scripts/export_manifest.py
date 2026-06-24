#!/usr/bin/env python3
"""Export collected_data/manifest.json to finetuning CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collection_dashboard.config import COLLECTION_OUTPUT_DIR  # noqa: E402
from collection_dashboard.server.manifest import ManifestStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export collected clips manifest CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=COLLECTION_OUTPUT_DIR / "collected_manifest.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    store = ManifestStore()
    snap = store.snapshot()
    rows: list[list[str | int]] = []
    for word, entry in snap["words"].items():
        for slot in entry["slots"]:
            if slot.get("status") != "complete" or not slot.get("file"):
                continue
            path = store.slot_path(word, slot["index"]).resolve()
            rows.append([str(path), word, entry["label_id"], "custom_train"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "label_id", "split"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
