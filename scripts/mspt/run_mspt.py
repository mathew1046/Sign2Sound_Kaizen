#!/usr/bin/env python3
"""Train Multi-Stream Pose Transformer (MSPT) on INCLUDE-50.

Memory-safe defaults for ~4 GB GPU / 8 GB RAM:
  --micro-batch-size 4 --grad-accum 8  -> effective batch 32
  --max-seq-len 96                       -> cap long clips via uniform subsample
  --amp                                  -> mixed precision on CUDA
  --num-workers 2                        -> parallel CPU load (mmap per sample)

Recommended CLI (smoke test):
  python run_mspt.py --epochs 2 --micro-batch-size 4 --grad-accum 2 --max-seq-len 96 --skip-body-wait

Full training:
  export INCLUDE50_LAB_ROOT=/path/to/include50_lab
  python run_mspt.py --epochs 150 --micro-batch-size 4 --grad-accum 8 --max-seq-len 96 --amp
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
from torch.utils.data import DataLoader

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

import slr_common as C  # noqa: E402
from mspt.collate import collate_mspt_batch  # noqa: E402
from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM, MSPTDataset  # noqa: E402
from mspt.rtmlib_dataset import MSPTRtmlibDataset  # noqa: E402
from mspt.pose_utils import body_ready  # noqa: E402
from mspt.model import MSPT  # noqa: E402


def _gpu_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def ensure_body_cache(lab_root: Path, force: bool = False, skip_wait: bool = False) -> None:
    """Ensure 33-pose body cache is complete before training."""
    body_dir = lab_root / "cache" / "mspt_body"
    if not force and body_ready(lab_root):
        n = sum(1 for _ in body_dir.rglob("*.npy"))
        print(f"[mspt] body cache ready ({n} files)")
        return
    if skip_wait:
        n = sum(1 for _ in body_dir.rglob("*.npy")) if body_dir.is_dir() else 0
        print(f"[mspt] WARN skip-body-wait: only {n} body files (upper-pose fallback enabled)")
        return

    from mspt.extract_body import process_manifest

    print("[mspt] extracting 33-pose body stream...")
    pose_model = C.INCLUDE_ML_ROOT / "models" / "pose_landmarker_full.task"
    body_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        mp = lab_root / "manifests" / f"{split}.csv"
        if mp.exists():
            process_manifest(
                mp, body_dir, pose_model, force=force,
                log_path=body_dir / f"failures_{split}.log",
            )
    n_body = sum(1 for _ in body_dir.rglob("*.npy"))
    n_lm = sum(1 for _ in (lab_root / "cache" / "landmarks").rglob("*.npy"))
    print(f"[mspt] body extraction done: {n_body}/{n_lm} -> {body_dir}")
    if not body_ready(lab_root):
        raise RuntimeError(f"Body cache incomplete ({n_body}/{n_lm}). Check failure logs in {body_dir}")


@torch.inference_mode()
def evaluate(model, loader, device, use_amp: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = total = 0
    all_pred, all_true = [], []
    amp_ctx = torch.autocast("cuda", enabled=use_amp and device.startswith("cuda"))
    for batch in loader:
        with amp_ctx:
            hand = batch["hand"].to(device, non_blocking=True)
            body = batch["body"].to(device, non_blocking=True)
            face = batch["face"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(hand, body, face, mask)
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        all_pred.append(pred.cpu().numpy())
        all_true.append(labels.cpu().numpy())
    return correct / max(1, total), np.concatenate(all_true), np.concatenate(all_pred)


def train_mspt(
    lab_root: Path,
    epochs: int = 150,
    micro_batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 20,
    label_smoothing: float = 0.05,
    aug_repeat: int = 8,
    max_seq_len: int = 96,
    num_workers: int = 2,
    use_amp: bool = True,
    use_checkpoint: bool = True,
    device: str = "cuda",
    ckpt_dir: Path | None = None,
    skip_body_wait: bool = False,
):
    ensure_body_cache(lab_root, skip_wait=skip_body_wait)
    use_full_body = body_ready(lab_root) and not skip_body_wait
    if use_full_body:
        print("[mspt] using full 33-pose body stream (no upper-pose fallback)")

    landmarks_dir = lab_root / "cache" / "landmarks"
    body_dir = lab_root / "cache" / "mspt_body"
    face_dir = lab_root / "cache" / "landmarks_face"
    manifests = lab_root / "manifests"
    ds_kw = dict(require_body=use_full_body)

    pin_memory = device.startswith("cuda")
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mspt_batch,
        persistent_workers=num_workers > 0,
    )

    train_ds = MSPTDataset(
        manifests / "train.csv", landmarks_dir, body_dir, face_dir,
        max_frames=max_seq_len, split="train", training=True, repeat=aug_repeat, **ds_kw,
    )
    val_ds = MSPTDataset(
        manifests / "val.csv", landmarks_dir, body_dir, face_dir,
        max_frames=max_seq_len, split="val", training=False, **ds_kw,
    )
    test_ds = MSPTDataset(
        manifests / "test.csv", landmarks_dir, body_dir, face_dir,
        max_frames=max_seq_len, split="test", training=False, **ds_kw,
    )
    eff_batch = micro_batch_size * grad_accum
    print(
        f"[mspt] train={len(train_ds)} (x{aug_repeat} aug) val={len(val_ds)} test={len(test_ds)} "
        f"micro_bs={micro_batch_size} accum={grad_accum} effective_bs={eff_batch} max_T={max_seq_len}"
    )

    train_dl = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_dl = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw)
    test_dl = DataLoader(test_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw)

    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=C.NUM_CLASSES,
        max_len=max_seq_len,
        use_checkpoint=use_checkpoint,
        sequential_streams=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))
    amp_enabled = use_amp and device.startswith("cuda")

    ckpt_dir = ckpt_dir or (MSPT_CHECKPOINTS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "mspt_best.pt"

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_val = 0.0
    stale = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl):
            amp_ctx = torch.autocast("cuda", enabled=amp_enabled)
            with amp_ctx:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                {"model": model.state_dict(), "val_acc": val_acc, "epoch": epoch + 1, "test_acc": None},
                best_path,
            )
        else:
            stale += 1
        print(
            f"[mspt] epoch {epoch+1}/{epochs} loss={train_loss:.3f} val_acc={val_acc:.3f} "
            f"best={best_val:.3f} lr={sched.get_last_lr()[0]:.2e}{mem_s}",
            flush=True,
        )
        gc.collect()
        if stale >= patience:
            print(f"[mspt] early stop at epoch {epoch+1}")
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_acc, y_true, y_pred = evaluate(model, val_dl, device, use_amp=amp_enabled)
    test_acc, yt, yp = evaluate(model, test_dl, device, use_amp=amp_enabled)

    train_eval_ds = MSPTDataset(
        manifests / "train.csv", landmarks_dir, body_dir, face_dir,
        max_frames=max_seq_len, split="train", training=False, **ds_kw,
    )
    train_eval_dl = DataLoader(
        train_eval_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw
    )
    train_acc, _, _ = evaluate(model, train_eval_dl, device, use_amp=amp_enabled)

    peak = _gpu_mem_mb()
    print(f"\n[mspt] FINAL train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}")
    print(f"[mspt] checkpoint: {best_path}")
    if peak is not None:
        print(f"[mspt] peak GPU memory: {peak:.0f} MB")

    per_class = {}
    for c in range(C.NUM_CLASSES):
        mask_c = yt == c
        if mask_c.sum() > 0:
            per_class[c] = float((yp[mask_c] == c).mean())
    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    print("[mspt] worst 5 classes (id, acc):", worst)
    return {"train": train_acc, "val": val_acc, "test": test_acc, "ckpt": str(best_path), "peak_gpu_mb": peak}


def _infer_num_classes(*datasets) -> int:
    mx = -1
    for ds in datasets:
        for *_, label in ds.rows:
            mx = max(mx, int(label))
    return mx + 1


def train_mspt_rtmlib(
    lab_root: Path,
    epochs: int = 150,
    micro_batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 20,
    label_smoothing: float = 0.05,
    aug_repeat: int = 8,
    max_seq_len: int = 96,
    num_workers: int = 2,
    use_amp: bool = True,
    use_checkpoint: bool = True,
    device: str = "cuda",
    ckpt_dir: Path | None = None,
    ckpt_name: str = "mspt_rtmlib_1080_best.pt",
    num_classes: int | None = None,
):
    """Train MSPT on rtmlib COCO-WholeBody caches (same model dims as MediaPipe MSPT)."""
    cache_root = lab_root / "cache"
    manifests = lab_root / "manifests"
    if not (manifests / "train.csv").is_file():
        raise FileNotFoundError(f"Missing manifests under {manifests}; run prepare_rtmlib_manifests.py")

    print("[mspt-rtmlib] using rtmlib body(17->33 pad) + face(68->72 pad) + hand streams")

    pin_memory = device.startswith("cuda")
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mspt_batch,
        persistent_workers=num_workers > 0,
    )

    train_ds = MSPTRtmlibDataset(
        manifests / "train.csv", cache_root,
        max_frames=max_seq_len, split="train", training=True, repeat=aug_repeat,
    )
    val_ds = MSPTRtmlibDataset(
        manifests / "val.csv", cache_root,
        max_frames=max_seq_len, split="val", training=False,
    )
    test_ds = MSPTRtmlibDataset(
        manifests / "test.csv", cache_root,
        max_frames=max_seq_len, split="test", training=False,
    )
    n_classes = num_classes or _infer_num_classes(train_ds, val_ds, test_ds)
    eff_batch = micro_batch_size * grad_accum
    print(
        f"[mspt-rtmlib] train={len(train_ds)} (x{aug_repeat} aug) val={len(val_ds)} test={len(test_ds)} "
        f"classes={n_classes} micro_bs={micro_batch_size} accum={grad_accum} "
        f"effective_bs={eff_batch} max_T={max_seq_len}"
    )

    train_dl = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_dl = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw)
    test_dl = DataLoader(test_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw)

    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=n_classes,
        max_len=max_seq_len,
        use_checkpoint=use_checkpoint,
        sequential_streams=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))
    amp_enabled = use_amp and device.startswith("cuda")

    ckpt_dir = ckpt_dir or (MSPT_CHECKPOINTS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / ckpt_name

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_val = 0.0
    stale = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl):
            amp_ctx = torch.autocast("cuda", enabled=amp_enabled)
            with amp_ctx:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    "test_acc": None,
                    "num_classes": n_classes,
                },
                best_path,
            )
        else:
            stale += 1
        print(
            f"[mspt-rtmlib] epoch {epoch+1}/{epochs} loss={train_loss:.3f} val_acc={val_acc:.3f} "
            f"best={best_val:.3f} lr={sched.get_last_lr()[0]:.2e}{mem_s}",
            flush=True,
        )
        gc.collect()
        if stale >= patience:
            print(f"[mspt-rtmlib] early stop at epoch {epoch+1}")
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_acc, y_true, y_pred = evaluate(model, val_dl, device, use_amp=amp_enabled)
    test_acc, yt, yp = evaluate(model, test_dl, device, use_amp=amp_enabled)

    train_eval_ds = MSPTRtmlibDataset(
        manifests / "train.csv", cache_root,
        max_frames=max_seq_len, split="train", training=False,
    )
    train_eval_dl = DataLoader(
        train_eval_ds, batch_size=micro_batch_size, shuffle=False, **loader_kw
    )
    train_acc, _, _ = evaluate(model, train_eval_dl, device, use_amp=amp_enabled)

    peak = _gpu_mem_mb()
    print(f"\n[mspt-rtmlib] FINAL train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}")
    print(f"[mspt-rtmlib] checkpoint: {best_path}")
    if peak is not None:
        print(f"[mspt-rtmlib] peak GPU memory: {peak:.0f} MB")

    per_class = {}
    for c in range(n_classes):
        mask_c = yt == c
        if mask_c.sum() > 0:
            per_class[c] = float((yp[mask_c] == c).mean())
    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    print("[mspt-rtmlib] worst 5 classes (id, acc):", worst)
    return {
        "train": train_acc,
        "val": val_acc,
        "test": test_acc,
        "ckpt": str(best_path),
        "peak_gpu_mb": peak,
        "num_classes": n_classes,
    }


def eval_mspt(
    lab_root: Path,
    checkpoint: Path,
    micro_batch_size: int = 4,
    max_seq_len: int = 96,
    num_workers: int = 2,
    use_amp: bool = True,
    use_checkpoint: bool = False,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate a saved checkpoint on train, val, and test (no augmentation)."""
    use_full_body = body_ready(lab_root)
    landmarks_dir = lab_root / "cache" / "landmarks"
    body_dir = lab_root / "cache" / "mspt_body"
    face_dir = lab_root / "cache" / "landmarks_face"
    manifests = lab_root / "manifests"
    ds_kw = dict(require_body=use_full_body)

    pin_memory = device.startswith("cuda")
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mspt_batch,
        persistent_workers=num_workers > 0,
    )

    splits = {}
    for name in ("train", "val", "test"):
        ds = MSPTDataset(
            manifests / f"{name}.csv",
            landmarks_dir,
            body_dir,
            face_dir,
            max_frames=max_seq_len,
            split=name,
            training=False,
            **ds_kw,
        )
        splits[name] = DataLoader(
            ds, batch_size=micro_batch_size, shuffle=False, **loader_kw
        )
        print(f"[mspt eval] {name}: {len(ds)} clips")

    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=C.NUM_CLASSES,
        max_len=max_seq_len,
        use_checkpoint=use_checkpoint,
        sequential_streams=True,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    print(f"[mspt eval] loaded {checkpoint} (epoch {epoch})")

    amp_enabled = use_amp and device.startswith("cuda")
    metrics: dict[str, float] = {}
    all_true, all_pred = None, None
    for name, loader in splits.items():
        acc, y_true, y_pred = evaluate(model, loader, device, use_amp=amp_enabled)
        metrics[name] = acc
        print(f"[mspt eval] {name} accuracy: {acc * 100:.2f}% ({int((y_pred == y_true).sum())}/{len(y_true)})")
        if name == "test":
            all_true, all_pred = y_true, y_pred

    if all_true is not None and all_pred is not None:
        per_class = {}
        for c in range(C.NUM_CLASSES):
            mask_c = all_true == c
            if mask_c.sum() > 0:
                per_class[c] = float((all_pred[mask_c] == c).mean())
        worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
        best = sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]
        print("[mspt eval] worst 5 class ids (id, acc):", worst)
        print("[mspt eval] best 5 class ids (id, acc):", best)

    print(
        f"\n[mspt eval] FINAL train={metrics['train']:.4f} "
        f"val={metrics['val']:.4f} test={metrics['test']:.4f}"
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Train MSPT (memory-safe defaults for 4 GB GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--micro-batch-size", type=int, default=4, help="GPU micro-batch; use grad-accum for effective batch")
    ap.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective_bs = micro_bs * accum)")
    ap.add_argument("--max-seq-len", type=int, default=96, help="Cap clip length via uniform subsample")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (2-4 recommended)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--aug-repeat", type=int, default=8, help="Stochastic aug multiplier (virtual epoch size)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision on CUDA")
    ap.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Activation checkpointing in transformers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force-body", action="store_true")
    ap.add_argument("--skip-body-wait", action="store_true", help="Train with partial body cache + upper-pose fallback")
    ap.add_argument("--rtmlib", action="store_true", help="Train on rtmlib COCO-WholeBody lab layout")
    ap.add_argument("--ckpt-name", default="mspt_best.pt", help="Checkpoint filename under checkpoints/")
    ap.add_argument("--num-classes", type=int, default=None, help="Output classes (default: infer from manifests)")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate checkpoint on train/val/test and exit")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for --eval-only")
    args = ap.parse_args()

    if args.lab_root:
        os.environ["INCLUDE50_LAB_ROOT"] = str(args.lab_root)
    lab_root = C._resolve_lab_root()
    print("LAB_ROOT:", lab_root)

    if args.eval_only:
        ckpt = args.checkpoint or (MSPT_CHECKPOINTS / "mspt_best.pt")
        eval_mspt(
            lab_root,
            ckpt.resolve(),
            micro_batch_size=args.micro_batch_size,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            use_amp=args.amp,
            use_checkpoint=args.checkpointing,
            device=args.device,
        )
        return

    if args.rtmlib:
        default_rtmlib_ckpt = "mspt_rtmlib_1080_best.pt"
        ckpt_name = args.ckpt_name if args.ckpt_name != "mspt_best.pt" else default_rtmlib_ckpt
        train_mspt_rtmlib(
            lab_root,
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            grad_accum=args.grad_accum,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            label_smoothing=args.label_smoothing,
            aug_repeat=args.aug_repeat,
            use_amp=args.amp,
            use_checkpoint=args.checkpointing,
            device=args.device,
            ckpt_name=ckpt_name,
            num_classes=args.num_classes,
        )
        return

    if args.force_body:
        ensure_body_cache(lab_root, force=True)

    train_mspt(
        lab_root,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        aug_repeat=args.aug_repeat,
        use_amp=args.amp,
        use_checkpoint=args.checkpointing,
        device=args.device,
        skip_body_wait=args.skip_body_wait,
    )


if __name__ == "__main__":
    main()
