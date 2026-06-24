#!/usr/bin/env python3
"""Evaluate MSPT — confusion matrix PNG + per-class analysis JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
ROOT = REPO_ROOT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

import slr_common as C  # noqa: E402
from run_mspt import evaluate  # noqa: E402
from mspt.collate import collate_mspt_batch  # noqa: E402
from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM, MSPTDataset  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from mspt.label_utils import label_names_for_checkpoint, num_classes_from_checkpoint  # noqa: E402
from mspt.rtmlib_dataset import MSPTRtmlibDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

EVALS_DIR = ROOT / "collection_dashboard" / "evals"
DEFAULT_CKPT = MSPT_CHECKPOINTS / "mspt_best.pt"
DEFAULT_RTMLIB_CKPT = MSPT_CHECKPOINTS / "mspt_rtmlib_263_best.pt"
DEFAULT_RTMLIB_LAB = RTMLIB_LAB
SPLIT_CHOICES = ("all", "train", "val", "test")


def _label_names(num_classes: int, lab_root: Path | None = None) -> list[str]:
    if num_classes > C.NUM_CLASSES and lab_root is not None:
        ckpt_stub = {"num_classes": num_classes}
        return label_names_for_checkpoint(ckpt_stub, lab_root=lab_root)
    _, idx_to_label = C.load_label_map()
    return [idx_to_label.get(i, f"class_{i}") for i in range(num_classes)]


def _short(name: str, n: int = 14) -> str:
    d = name.replace("_", " ")
    return d if len(d) <= n else d[: n - 1] + "…"


def _manifest_path(lab_root: Path, split: str) -> Path:
    if split == "all":
        all_path = lab_root / "manifests" / "all.csv"
        if all_path.is_file():
            return all_path
        parts = []
        for name in ("train", "val", "test"):
            p = lab_root / "manifests" / f"{name}.csv"
            if p.is_file():
                parts.append(pd.read_csv(p))
        if not parts:
            raise FileNotFoundError(f"No manifests under {lab_root / 'manifests'}")
        combined = pd.concat(parts, ignore_index=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        combined.to_csv(tmp.name, index=False)
        return Path(tmp.name)
    path = lab_root / "manifests" / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path


@torch.inference_mode()
def run_eval_rtmlib(
    lab_root: Path,
    checkpoint: Path,
    split: str,
    device: str,
    micro_batch_size: int = 4,
    max_seq_len: int = 96,
    num_workers: int = 0,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    manifest = _manifest_path(lab_root, split)
    cache_root = lab_root / "cache"
    ds = MSPTRtmlibDataset(
        manifest,
        cache_root,
        max_frames=max_seq_len,
        training=False,
    )
    n_clips = len(ds)
    print(f"[eval-rtmlib] split={split} clips={n_clips} manifest={manifest.name}")
    loader = DataLoader(
        ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mspt_batch,
        pin_memory=device.startswith("cuda"),
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    n_classes = num_classes_from_checkpoint(ckpt) or C.NUM_CLASSES
    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=n_classes,
        max_len=max_seq_len,
        use_checkpoint=False,
        sequential_streams=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    acc, y_true, y_pred = evaluate(model, loader, device, use_amp=use_amp and device.startswith("cuda"))
    return y_true, y_pred, acc, n_clips, n_classes


@torch.inference_mode()
def run_eval(
    lab_root: Path,
    checkpoint: Path,
    split: str,
    device: str,
    micro_batch_size: int = 4,
    max_seq_len: int = 96,
    num_workers: int = 0,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    landmarks_dir = lab_root / "cache" / "landmarks"
    body_dir = lab_root / "cache" / "mspt_body"
    face_dir = lab_root / "cache" / "landmarks_face"
    manifest = _manifest_path(lab_root, split)
    ds = MSPTDataset(
        manifest,
        landmarks_dir,
        body_dir,
        face_dir,
        max_frames=max_seq_len,
        training=False,
        require_body=False,
    )
    n_clips = len(ds)
    print(f"[eval] split={split} clips={n_clips} manifest={manifest.name}")
    loader = DataLoader(
        ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mspt_batch,
        pin_memory=device.startswith("cuda"),
    )
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
    acc, y_true, y_pred = evaluate(model, loader, device, use_amp=use_amp and device.startswith("cuda"))
    return y_true, y_pred, acc, n_clips, C.NUM_CLASSES


def build_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
    split: str,
    n_clips: int,
    checkpoint: Path,
    num_classes: int,
    lab_root: Path | None = None,
) -> dict:
    names = _label_names(num_classes, lab_root)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class: list[dict] = []
    for c in range(num_classes):
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((y_pred[mask] == c).sum())
        acc = correct / n
        row = cm[c]
        confused_with: list[dict] = []
        for other, cnt in enumerate(row):
            if other != c and cnt > 0:
                confused_with.append(
                    {"word": names[other], "label_id": other, "count": int(cnt)}
                )
        confused_with.sort(key=lambda x: -x["count"])
        per_class.append(
            {
                "label_id": c,
                "word": names[c],
                "display_name": names[c].replace("_", " ").title(),
                "n_clips": n,
                "n_test": n,  # legacy field for dashboard
                "correct": correct,
                "accuracy": round(acc, 4),
                "top_confusions": confused_with[:5],
            }
        )
    per_class.sort(key=lambda x: x["accuracy"])
    worst = per_class[:10]
    best = sorted(per_class, key=lambda x: -x["accuracy"])[:10]

    check_set: dict[int, dict] = {}
    for entry in worst[:8]:
        check_set[entry["label_id"]] = {
            "word": entry["word"],
            "reason": f"low accuracy ({entry['accuracy'] * 100:.1f}%)",
            "accuracy": entry["accuracy"],
        }
    for entry in per_class:
        if entry["top_confusions"] and entry["top_confusions"][0]["count"] >= 2:
            lid = entry["label_id"]
            if lid not in check_set:
                check_set[lid] = {
                    "word": entry["word"],
                    "reason": f"often confused with {entry['top_confusions'][0]['word']}",
                    "accuracy": entry["accuracy"],
                }
    check_list = sorted(
        [{"label_id": k, **v} for k, v in check_set.items()],
        key=lambda x: x["accuracy"],
    )

    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "n_clips": n_clips,
        "accuracy": round(accuracy, 6),
        "test_accuracy": round(accuracy, 6),  # legacy field for dashboard
        "num_classes": num_classes,
        "best_classes": best,
        "worst_classes": worst,
        "check_skeleton_classes": check_list,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "label_names": names,
    }


def save_confusion_png(
    cm: np.ndarray,
    names: list[str],
    out: Path,
    split: str,
    checkpoint: Path,
    num_classes: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(14, num_classes * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    short = [_short(n) for n in names]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    split_label = "full dataset" if split == "all" else split
    ax.set_title(f"MSPT confusion matrix — {split_label} ({checkpoint.name})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="MSPT confusion matrix + class analysis")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument(
        "--rtmlib",
        action="store_true",
        help="Evaluate rtmlib COCO-WholeBody lab (default ckpt: mspt_rtmlib_1080_best.pt)",
    )
    ap.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="all",
        help="Dataset split to evaluate (default: all = entire INCLUDE-50)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=EVALS_DIR)
    args = ap.parse_args()

    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    if args.rtmlib:
        lab_root = Path(args.lab_root or DEFAULT_RTMLIB_LAB)
        ckpt = (args.checkpoint or DEFAULT_RTMLIB_CKPT).resolve()
    else:
        lab_root = C._resolve_lab_root()
        ckpt = (args.checkpoint or DEFAULT_CKPT).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    print(f"[eval] lab={lab_root} ckpt={ckpt} rtmlib={args.rtmlib}")
    if args.rtmlib:
        y_true, y_pred, acc, n_clips, n_classes = run_eval_rtmlib(
            lab_root, ckpt, args.split, args.device,
        )
    else:
        y_true, y_pred, acc, n_clips, n_classes = run_eval(
            lab_root, ckpt, args.split, args.device,
        )
    analysis = build_analysis(
        y_true, y_pred, acc, args.split, n_clips, ckpt, n_classes, lab_root if args.rtmlib else None,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_name = "confusion_matrix_all.png" if args.split == "all" else f"confusion_matrix_{args.split}.png"
    png_path = out_dir / png_name
    json_path = out_dir / "eval_analysis.json"
    cm = np.array(analysis["confusion_matrix"], dtype=int)
    save_confusion_png(cm, analysis["label_names"], png_path, args.split, ckpt, n_classes)
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    # Keep legacy filename for dashboard when evaluating full dataset
    if args.split == "all":
        legacy = out_dir / "confusion_matrix_test.png"
        legacy.write_bytes(png_path.read_bytes())

    split_label = "full dataset" if args.split == "all" else args.split
    print(f"[eval] {split_label} accuracy: {acc * 100:.2f}% ({n_clips} clips)")
    print(f"[eval] saved {png_path}")
    print(f"[eval] saved {json_path}")
    if analysis["best_classes"]:
        b = analysis["best_classes"][0]
        print(f"[eval] best:  {b['word']} ({b['accuracy'] * 100:.1f}%)")
    if analysis["worst_classes"]:
        w = analysis["worst_classes"][0]
        print(f"[eval] worst: {w['word']} ({w['accuracy'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
