"""
Training Pipeline for BiLSTM Sign Language Recognition

This script implements the complete training pipeline including:
- Data loading from preprocessed features
- Model training with validation
- Learning rate scheduling
- Early stopping
- Checkpointing
- Metric logging and visualization

Usage:
    python training/train.py --config training/config.yaml [--device cuda] [--resume checkpoint.pth]

Author: Team Kaizen
Date: January 2026
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import BiLSTMClassifier, BiLSTMCTCClassifier
from models.loss import calculate_class_weights, get_loss_function
from training.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """Dataset for sign language features."""

    def __init__(self, csv_path: str, features_dir: str):
        """
        Initialize dataset.

        Args:
            csv_path: Path to split CSV file
            features_dir: Directory containing .npy feature files
        """
        self.data = pd.read_csv(csv_path)
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load features
        feature_file = self.features_dir / row["file_path"].split("/")[-1]
        features = np.load(str(feature_file))

        # Ensure shape (60, 126)
        if features.ndim == 1:
            padded = np.zeros((60, 126), dtype=np.float32)
            padded[0] = features
            features = padded

        # Get sequence length
        seq_length = min(int(row["seq_length"]), 60) if "seq_length" in row else 60

        return {
            "features": torch.from_numpy(features).float(),
            "label": int(row["class_idx"]),
            "seq_length": seq_length,
            "sample_id": row["sample_id"],
        }


def collate_fn(batch):
    """Custom collate function for batching with padding for variable lengths."""
    # Get all features and find max sequence length in batch
    features_list = [item["features"] for item in batch]
    seq_lengths = [item["seq_length"] for item in batch]

    # Find max sequence length in this batch
    max_len = max(f.size(0) for f in features_list)
    feature_dim = features_list[0].size(1)

    # Pad all sequences to max_len
    padded_features = []
    for features in features_list:
        curr_len = features.size(0)
        if curr_len < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - curr_len, feature_dim, dtype=features.dtype)
            padded = torch.cat([features, padding], dim=0)
            padded_features.append(padded)
        else:
            padded_features.append(features)

    # Stack padded features
    features = torch.stack(padded_features)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

    return {"features": features, "labels": labels, "seq_lengths": seq_lengths}


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
    use_ctc: bool = False,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        gradient_clip: Gradient clipping value
        use_ctc: Whether using CTC loss

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        seq_lengths = batch["seq_lengths"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(features, seq_lengths)

        if use_ctc:
            # CTC loss expects (T, batch, num_classes)
            # and needs output lengths (computed from input sequence lengths and downsampling)
            # For LSTM without downsampling, output_lengths = input_lengths
            output_lengths = seq_lengths

            # Create target sequences: repeat each label once (since input is one sequence per class)
            # For CTC, we need target sequences. Using single class label per sequence.
            targets = labels.unsqueeze(1)  # (batch,) -> (batch, 1)
            target_lengths = torch.ones_like(labels)  # Each target is single label

            loss = criterion(logits, targets, output_lengths, target_lengths)

            # For accuracy, take argmax of CTC outputs (sequence mean)
            seq_probs = torch.softmax(logits, dim=2)  # (T, batch, num_classes)
            predictions = seq_probs.mean(dim=0).argmax(dim=1)  # (batch,)
        else:
            # Standard loss expects (batch, num_classes) and (batch,)
            loss = criterion(logits, labels)
            _, predictions = torch.max(logits, 1)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            {"loss": total_loss / (total / 32), "acc": 100 * correct / total}
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_ctc: bool = False,
) -> Tuple[float, float]:
    """
    Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        use_ctc: Whether using CTC loss

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")

        for batch in progress_bar:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)

            logits = model(features, seq_lengths)

            if use_ctc:
                output_lengths = seq_lengths
                targets = labels.unsqueeze(1)
                target_lengths = torch.ones_like(labels)
                loss = criterion(logits, targets, output_lengths, target_lengths)

                # For accuracy, take argmax of CTC outputs (sequence mean)
                seq_probs = torch.softmax(logits, dim=2)  # (T, batch, num_classes)
                predictions = seq_probs.mean(dim=0).argmax(dim=1)  # (batch,)
            else:
                loss = criterion(logits, labels)
                _, predictions = torch.max(logits, 1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(
                {"loss": total_loss / (total / 32), "acc": 100 * correct / total}
            )

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(config: Dict, device: str = None, resume_from: str = None):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        device: Device to use (auto-detect if None)
        resume_from: Path to checkpoint to resume from
    """
    # Setup device
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directories
    checkpoint_dir = Path(config["checkpointing"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(config["logging"]["log_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading datasets...")
    train_dataset = SignLanguageDataset(
        config["data"]["train_csv"],
        Path(config["data"]["train_csv"]).parent / "features",
    )
    val_dataset = SignLanguageDataset(
        config["data"]["val_csv"], Path(config["data"]["val_csv"]).parent / "features"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["data"]["shuffle_train"],
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["data"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["data"]["pin_memory"],
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model based on loss type
    logger.info("Creating model...")
    use_ctc = config["training"]["loss_type"].lower() == "ctc"

    if use_ctc:
        logger.info("Using BiLSTM CTC model")
        model = BiLSTMCTCClassifier(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    else:
        logger.info("Using standard BiLSTM model")
        model = BiLSTMClassifier(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    model = model.to(device)
    logger.info(f"Loss type: {config['training']['loss_type']}")

    # Calculate class weights (skip for CTC)
    train_csv = config["data"]["train_csv"]
    if not use_ctc:
        class_weights = calculate_class_weights(
            train_csv, num_classes=config["model"]["num_classes"], device=device
        )
    else:
        class_weights = None
        logger.info("CTC loss does not use class weights")

    # Loss function
    criterion = get_loss_function(
        config["training"]["loss_type"],
        class_weights=class_weights
        if config["training"]["use_class_weights"]
        else None,
        num_classes=config["model"]["num_classes"],
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
        min_lr=config["scheduler"]["min_lr"],
    )

    # Callbacks
    early_stopping = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
        verbose=True,
    )

    checkpoint = ModelCheckpoint(checkpoint_dir=str(checkpoint_dir), verbose=True)

    # Training loop
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_from:
        checkpoint_data = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        start_epoch = checkpoint_data["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config["training"]["gradient_clip"],
            use_ctc=use_ctc,
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        if (epoch + 1) % config["training"]["val_every"] == 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, use_ctc=use_ctc
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Checkpointing
            checkpoint.update(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                config=config["model"],
                model_type="bilstm_ctc" if use_ctc else "bilstm",
                loss_type=config["training"]["loss_type"],
            )

            # Early stopping
            early_stopping.update(val_loss)
            if early_stopping.should_stop():
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Save final model
    if config["checkpointing"]["save_final"]:
        final_path = checkpoint_dir / "final_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config["model"],
                "loss_type": config["training"]["loss_type"],
                "model_type": "bilstm_ctc" if use_ctc else "bilstm",
                "train_loss": train_loss,
                "val_loss": val_losses[-1] if val_losses else 0,
                "train_acc": train_acc,
                "val_acc": val_accs[-1] if val_accs else 0,
            },
            final_path,
        )
        logger.info(f"Final model saved to {final_path}")

    # Save metrics
    metrics = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }

    metrics_path = results_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump({k: [float(v) for v in vs] for k, vs in metrics.items()}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plot_path = results_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Training curves saved to {plot_path}")
    plt.close()

    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Train BiLSTM model for sign language recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Train
    train_model(config, device=args.device, resume_from=args.resume)


if __name__ == "__main__":
    main()
