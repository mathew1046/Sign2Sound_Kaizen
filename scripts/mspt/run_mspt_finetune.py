#!/usr/bin/env python3
"""Fine-tune MSPT from mspt_best.pt on collected webcam data (local npy caches).

Defaults tuned for 4 GB VRAM / 8 GB RAM:
  micro_bs=2, grad_accum=16, num_workers=0, max_seq_len=96, amp + activation checkpointing

Train data: collected_data/cache/ + manifest.csv (webcam clips, un-mirrored at preprocess).
Val data:   original include50_lab val split (unchanged baseline metric).

Smoke test:
  python collected_data/preprocess_collected.py --limit 4
  python run_mspt_finetune.py --smoke-test
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

NOTEBOOKS = Path(__file__).resolve().parent
ROOT = REPO_ROOT
COLLECTED_DEFAULT = ROOT / "collected_data"
sys.path.insert(0, str(NOTEBOOKS))

import slr_common as C  # noqa: E402
from mspt.collate import collate_mspt_batch  # noqa: E402
from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM, MSPTDataset  # noqa: E402
from mspt.extract_body import body_ready  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from run_mspt import _gpu_mem_mb, evaluate  # noqa: E402


def _collected_cache(collected_dir: Path) -> tuple[Path, Path, Path, Path]:
    cache = collected_dir / "cache"
    manifest = collected_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}. Run: python collected_data/preprocess_collected.py --export-only"
        )
    return (
        manifest,
        cache / "landmarks",
        cache / "mspt_body",
        cache / "landmarks_face",
    )


def apply_freeze(model: MSPT, mode: str) -> None:
    """Limit trainable params for low-data finetuning."""
    if mode == "none":
        return
    for p in model.parameters():
        p.requires_grad = False
    if mode == "encoders":
        for mod in (model.fusion, model.joint_layers, model.classifier):
            for p in mod.parameters():
                p.requires_grad = True
        model.joint_cls.requires_grad = True
    elif mode == "classifier_only":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif mode == "face_fusion":
        for mod in (model.face_enc, model.fusion, model.joint_layers, model.classifier):
            for p in mod.parameters():
                p.requires_grad = True
        model.joint_cls.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode: {mode!r}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] freeze_mode={mode} trainable={n_train}/{n_total} params")


def finetune_mspt(
    lab_root: Path,
    collected_dir: Path,
    init_ckpt: Path,
    epochs: int = 30,
    micro_batch_size: int = 2,
    grad_accum: int = 16,
    lr: float = 3e-5,
    weight_decay: float = 1e-3,
    patience: int = 10,
    label_smoothing: float = 0.05,
    aug_repeat: int = 2,
    max_seq_len: int = 96,
    num_workers: int = 0,
    use_amp: bool = True,
    use_checkpoint: bool = True,
    device: str = "cuda",
    ckpt_dir: Path | None = None,
    merge_original_train: bool = False,
    freeze_mode: str = "none",
    grad_clip: float = 1.0,
):
    train_manifest, c_lm, c_body, c_face = _collected_cache(collected_dir)
    n_collected = sum(1 for _ in c_lm.rglob("*.npy")) if c_lm.is_dir() else 0
    if n_collected == 0:
        raise RuntimeError(
            f"No npy in {c_lm}. Run: python collected_data/preprocess_collected.py"
        )

    lab_lm = lab_root / "cache" / "landmarks"
    lab_body = lab_root / "cache" / "mspt_body"
    lab_face = lab_root / "cache" / "landmarks_face"
    manifests = lab_root / "manifests"
    use_full_body = body_ready(lab_root)
    ds_kw = dict(require_body=use_full_body, max_frames=max_seq_len)

    collected_ds = MSPTDataset(
        train_manifest, c_lm, c_body, c_face,
        split="train", training=True, repeat=aug_repeat, **ds_kw,
    )
    if len(collected_ds) == 0:
        raise RuntimeError(f"Collected dataset empty — check caches under {collected_dir / 'cache'}")

    if merge_original_train:
        orig_ds = MSPTDataset(
            manifests / "train.csv", lab_lm, lab_body, lab_face,
            split="train", training=True, repeat=1, **ds_kw,
        )
        train_ds: ConcatDataset | MSPTDataset = ConcatDataset([orig_ds, collected_ds])
        train_label = f"orig({len(orig_ds)})+collected({len(collected_ds)})"
    else:
        train_ds = collected_ds
        train_label = f"collected({len(collected_ds)})"

    val_ds = MSPTDataset(
        manifests / "val.csv", lab_lm, lab_body, lab_face,
        split="val", training=False, **ds_kw,
    )

    pin_memory = device.startswith("cuda")
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mspt_batch,
        persistent_workers=num_workers > 0,
    )
    eff_batch = micro_batch_size * grad_accum
    print(
        f"[finetune] train={train_label} x{aug_repeat if not merge_original_train else 'mixed'} "
        f"val={len(val_ds)} micro_bs={micro_batch_size} accum={grad_accum} "
        f"effective_bs={eff_batch} max_T={max_seq_len}"
    )
    print(f"[finetune] collected cache: {collected_dir / 'cache'}")
    print(f"[finetune] init checkpoint: {init_ckpt}")

    train_dl = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=len(train_ds) >= micro_batch_size,
        **loader_kw,
    )
    val_dl = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw)

    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=C.NUM_CLASSES,
        max_len=max_seq_len,
        use_checkpoint=use_checkpoint,
        sequential_streams=True,
    ).to(device)

    ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    baseline_val = ckpt.get("val_acc")
    print(f"[finetune] loaded weights (baseline val_acc={baseline_val})")
    apply_freeze(model, freeze_mode)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))
    amp_enabled = use_amp and device.startswith("cuda")

    ckpt_dir = ckpt_dir or (MSPT_CHECKPOINTS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "mspt_finetuned.pt"

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_val = 0.0
    stale = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl):
            with torch.autocast("cuda", enabled=amp_enabled):
                hand = batch["hand"].to(device, non_blocking=True)
                body = batch["body"].to(device, non_blocking=True)
                face = batch["face"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                logits = model(hand, body, face, mask)
                loss = crit(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            del hand, body, face, mask, labels, logits, loss

        sched.step()
        val_acc, _, _ = evaluate(model, val_dl, device, use_amp=amp_enabled)
        train_loss = running / max(1, len(train_dl))
        mem = _gpu_mem_mb()
        mem_s = f" peak_gpu={mem:.0f}MB" if mem is not None else ""
        if val_acc > best_val:
            best_val = val_acc
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "init_ckpt": str(init_ckpt),
                    "collected_dir": str(collected_dir),
                },
                best_path,
            )
        else:
            stale += 1
        print(
            f"[finetune] epoch {epoch+1}/{epochs} loss={train_loss:.3f} val_acc={val_acc:.3f} "
            f"best={best_val:.3f} lr={sched.get_last_lr()[0]:.2e}{mem_s}",
            flush=True,
        )
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if stale >= patience:
            print(f"[finetune] early stop at epoch {epoch+1}")
            break

    peak = _gpu_mem_mb()
    print(f"\n[finetune] FINAL best_val={best_val:.3f} checkpoint={best_path}")
    if peak is not None:
        print(f"[finetune] peak GPU memory: {peak:.0f} MB")
    return {"val_acc": best_val, "ckpt": str(best_path), "peak_gpu_mb": peak}


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune MSPT on collected webcam data (4 GB VRAM defaults).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--collected-dir", type=Path, default=COLLECTED_DEFAULT)
    ap.add_argument("--init-checkpoint", type=Path, default=MSPT_CHECKPOINTS / "mspt_best.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--aug-repeat", type=int, default=2)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge-original-train", action="store_true",
                    help="Concat original include50 train split with collected clips")
    ap.add_argument("--freeze-mode", default="none",
                    choices=("none", "encoders", "classifier_only", "face_fusion"))
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--smoke-test", action="store_true",
                    help="2 epochs, patience=99 — for pipeline verification")
    args = ap.parse_args()

    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    lab_root = C._resolve_lab_root()
    print("LAB_ROOT:", lab_root)

    epochs = 2 if args.smoke_test else args.epochs
    patience = 99 if args.smoke_test else args.patience

    finetune_mspt(
        lab_root,
        args.collected_dir.resolve(),
        args.init_checkpoint.resolve(),
        epochs=epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=patience,
        label_smoothing=args.label_smoothing,
        aug_repeat=args.aug_repeat,
        use_amp=args.amp,
        use_checkpoint=args.checkpointing,
        device=args.device,
        merge_original_train=args.merge_original_train,
        freeze_mode=args.freeze_mode,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
