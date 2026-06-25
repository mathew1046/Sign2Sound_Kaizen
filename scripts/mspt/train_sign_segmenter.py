#!/usr/bin/env python3
"""Train frame-level sign/idle segmenter on synthetic continuous sessions.

Usage:
  python scripts/mspt/build_continuous_rtmlib_sessions.py
  python scripts/mspt/train_sign_segmenter.py \\
    --continuous-dir data/include50_rtmlib_1080/continuous \\
    --out checkpoints/mspt/sign_segmenter_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mspt.segmentation import wholebody_frame_features  # noqa: E402
from mspt.segmenter_model import DEFAULT_INPUT_DIM, SignFrameClassifier  # noqa: E402


class ContinuousSegmentDataset(Dataset):
    def __init__(self, continuous_dir: Path, split: str, max_frames: int = 256):
        self.continuous_dir = Path(continuous_dir)
        manifest = pd.read_csv(self.continuous_dir / "manifest.csv")
        self.rows = manifest[manifest["split"] == split].to_dict("records")
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        sid = self.rows[idx]["session_id"]
        wb = np.load(self.continuous_dir / f"{sid}_wholebody.npy").astype(np.float32)
        labels = np.load(self.continuous_dir / f"{sid}_label.npy").astype(np.float32)
        t = min(len(wb), len(labels), self.max_frames)
        feats = np.stack([wholebody_frame_features(wb[i]) for i in range(t)], axis=0)
        y = labels[:t]
        return {
            "feats": torch.from_numpy(feats),
            "labels": torch.from_numpy(y),
            "length": t,
        }


def collate(batch: list[dict]) -> dict:
    max_t = max(b["length"] for b in batch)
    d = DEFAULT_INPUT_DIM
    feats = torch.zeros(len(batch), max_t, d)
    labels = torch.zeros(len(batch), max_t)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["length"]
        feats[i, :t] = b["feats"]
        labels[i, :t] = b["labels"]
        lengths[i] = t
    return {"feats": feats, "labels": labels, "lengths": lengths}


@torch.no_grad()
def eval_metrics(model: SignFrameClassifier, loader: DataLoader, device: str) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        feats = batch["feats"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        logits = model(feats, lengths.to(device))
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        for i in range(feats.size(0)):
            t = int(lengths[i])
            p = preds[i, :t]
            y = labels[i, :t]
            tp += int(((p == 1) & (y == 1)).sum())
            fp += int(((p == 1) & (y == 0)).sum())
            fn += int(((p == 0) & (y == 1)).sum())
            tn += int(((p == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}


def train(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_ds = ContinuousSegmentDataset(args.continuous_dir, "train", args.max_frames)
    val_ds = ContinuousSegmentDataset(args.continuous_dir, "val", args.max_frames)
    if len(train_ds) == 0:
        raise SystemExit(f"No training sessions in {args.continuous_dir}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    ) if len(val_ds) else None

    # Estimate pos_weight from a sample
    pos, neg = 0.0, 0.0
    for i in range(min(20, len(train_ds))):
        item = train_ds[i]
        y = item["labels"].numpy()
        pos += float(y.sum())
        neg += float(len(y) - y.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], device=device)

    model = SignFrameClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            feats = batch["feats"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(feats, lengths)
            mask = torch.arange(feats.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        msg = f"epoch {epoch} loss={total_loss / max(1, n_batches):.4f}"
        if val_loader is not None:
            metrics = eval_metrics(model, val_loader, device)
            msg += f" val_f1={metrics['f1']:.3f} val_acc={metrics['accuracy']:.3f}"
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "input_dim": DEFAULT_INPUT_DIM,
                        "val_f1": metrics["f1"],
                        "epoch": epoch,
                    },
                    args.out,
                )
                print(f"{msg}  [saved {args.out}]")
            else:
                print(msg)
        else:
            print(msg)

    if val_loader is None:
        torch.save(
            {"model": model.state_dict(), "input_dim": DEFAULT_INPUT_DIM, "epoch": args.epochs},
            args.out,
        )
        print(f"[train] saved {args.out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train sign/idle frame segmenter")
    ap.add_argument(
        "--continuous-dir",
        type=Path,
        default=REPO_ROOT / "data" / "include50_rtmlib_1080" / "continuous",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "mspt" / "sign_segmenter_best.pt",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
