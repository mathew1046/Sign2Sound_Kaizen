#!/usr/bin/env python3
"""Evaluate and compare MSPT checkpoints (baseline vs finetuned).

Reports accuracy on:
  - INCLUDE-50 val / test (original lab caches)
  - collected webcam clips (collected_data/cache/)

Usage (conda base):
  cd notebooks
  python test_mspt_finetuned.py

  python test_mspt_finetuned.py \\
    --baseline checkpoints/mspt_best.pt \\
    --finetuned checkpoints/mspt_finetuned.pt

  python test_mspt_finetuned.py --finetuned checkpoints/mspt_finetuned.pt --collected-only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NOTEBOOKS = Path(__file__).resolve().parent
ROOT = REPO_ROOT
COLLECTED_DEFAULT = ROOT / "collected_data"
sys.path.insert(0, str(NOTEBOOKS))

import slr_common as C  # noqa: E402
from mspt.collate import collate_mspt_batch  # noqa: E402
from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM, MSPTDataset  # noqa: E402
from mspt.extract_body import body_ready  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from run_mspt import evaluate  # noqa: E402


def load_label_names() -> dict[int, str]:
    words_csv = ROOT / "include50_words.csv"
    if not words_csv.is_file():
        return {}
    names: dict[int, str] = {}
    with words_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[int(row["label_id"])] = row["word"].strip()
    return names


def load_mspt(checkpoint: Path, max_seq_len: int, device: str) -> MSPT:
    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=C.NUM_CLASSES,
        max_len=max_seq_len,
        use_checkpoint=False,
        sequential_streams=True,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def eval_split(
    model: MSPT,
    manifest: Path,
    lm_dir: Path,
    body_dir: Path,
    face_dir: Path,
    split: str | None,
    device: str,
    micro_batch_size: int,
    max_seq_len: int,
    require_body: bool,
    num_workers: int = 0,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    ds = MSPTDataset(
        manifest, lm_dir, body_dir, face_dir,
        max_frames=max_seq_len,
        split=split,
        training=False,
        require_body=require_body,
    )
    if len(ds) == 0:
        return 0.0, np.array([]), np.array([]), 0
    dl = DataLoader(
        ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_mspt_batch,
    )
    acc, y_true, y_pred = evaluate(model, dl, device, use_amp=device.startswith("cuda"))
    return acc, y_true, y_pred, len(ds)


def per_word_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict[int, str],
) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for label_id in sorted(set(y_true.tolist())):
        mask = y_true == label_id
        acc = float((y_pred[mask] == label_id).mean()) if mask.sum() else 0.0
        name = label_names.get(int(label_id), str(label_id))
        rows.append((name, int(mask.sum()), acc))
    return sorted(rows, key=lambda x: x[2])


def print_per_word(rows: list[tuple[str, int, float]], title: str, worst_n: int = 10) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no clips)")
        return
    overall = sum(a * n for _, n, a in rows) / max(1, sum(n for _, n, _ in rows))
    print(f"  overall: {overall * 100:.1f}%  ({sum(n for _, n, _ in rows)} clips, {len(rows)} words)")
    print(f"  worst {worst_n} words:")
    for name, n, acc in rows[:worst_n]:
        print(f"    {name:20s}  {acc * 100:5.1f}%  (n={n})")


def run_eval(
    name: str,
    checkpoint: Path,
    lab_root: Path,
    collected_dir: Path,
    device: str,
    micro_batch_size: int,
    max_seq_len: int,
    include_lab: bool,
    include_collected: bool,
    label_names: dict[int, str],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Checkpoint: {name}")
    print(f"  path: {checkpoint}")
    model, meta = load_mspt(checkpoint, max_seq_len, device)
    val_acc = meta.get("val_acc")
    val_s = f"saved_val_acc={val_acc:.4f}" if isinstance(val_acc, float) else "saved_val_acc=?"
    print(f"  meta: epoch={meta.get('epoch', '?')}  {val_s}")

    require_body = body_ready(lab_root)
    lab_lm = lab_root / "cache" / "landmarks"
    lab_body = lab_root / "cache" / "mspt_body"
    lab_face = lab_root / "cache" / "landmarks_face"
    manifests = lab_root / "manifests"

    if include_lab:
        for split in ("val", "test"):
            mp = manifests / f"{split}.csv"
            if not mp.is_file():
                print(f"  [skip] missing {mp}")
                continue
            acc, y_true, y_pred, n = eval_split(
                model, mp, lab_lm, lab_body, lab_face, split,
                device, micro_batch_size, max_seq_len, require_body,
            )
            print(f"  include50 {split:5s}: {acc * 100:.2f}%  ({int((y_pred == y_true).sum())}/{n})")

    if include_collected:
        c_manifest = collected_dir / "manifest.csv"
        cache = collected_dir / "cache"
        acc, y_true, y_pred, n = eval_split(
            model, c_manifest,
            cache / "landmarks", cache / "mspt_body", cache / "landmarks_face",
            None, device, micro_batch_size, max_seq_len, require_body=False,
        )
        print(f"  collected      : {acc * 100:.2f}%  ({int((y_pred == y_true).sum()) if n else 0}/{n})")
        rows = per_word_accuracy(y_true, y_pred, label_names)
        print_per_word(rows, f"  per-word (collected) — {name}", worst_n=10)


def main():
    ap = argparse.ArgumentParser(
        description="Compare MSPT baseline vs finetuned checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--collected-dir", type=Path, default=COLLECTED_DEFAULT)
    ap.add_argument("--baseline", type=Path, default=MSPT_CHECKPOINTS / "mspt_best.pt")
    ap.add_argument("--finetuned", type=Path, default=MSPT_CHECKPOINTS / "mspt_finetuned.pt")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--finetuned-only", action="store_true")
    ap.add_argument("--collected-only", action="store_true", help="Skip INCLUDE-50 val/test")
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    lab_root = C._resolve_lab_root()
    label_names = load_label_names()
    include_lab = not args.collected_only
    run_baseline = not args.finetuned_only
    run_finetuned = not args.baseline_only

    print("LAB_ROOT:", lab_root)
    print("COLLECTED:", args.collected_dir.resolve())

    if run_baseline:
        if not args.baseline.is_file():
            print(f"WARNING: baseline checkpoint missing: {args.baseline}")
        else:
            run_eval(
                "baseline (mspt_best)", args.baseline.resolve(),
                lab_root, args.collected_dir, args.device,
                args.micro_batch_size, args.max_seq_len,
                include_lab, True, label_names,
            )

    if run_finetuned:
        if not args.finetuned.is_file():
            print(f"\nWARNING: finetuned checkpoint missing: {args.finetuned}")
            print("  Run: python run_mspt_finetune.py  or  python autoresearch_mspt/train.py")
        else:
            run_eval(
                "finetuned (mspt_finetuned)", args.finetuned.resolve(),
                lab_root, args.collected_dir, args.device,
                args.micro_batch_size, args.max_seq_len,
                include_lab, True, label_names,
            )

    print(f"\n{'=' * 60}")
    print("Live webcam test (2.5s clips, top-3 top-right, skeleton bottom-left):")
    print("  cd notebooks && python run_mspt_webcam_finetuned.py")
    print(f"  cd notebooks && python run_mspt_webcam_finetuned.py --checkpoint {args.finetuned}")


if __name__ == "__main__":
    main()
