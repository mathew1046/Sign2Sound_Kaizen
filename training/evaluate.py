"""
Model Evaluation Script

Evaluate trained model on test set and generate comprehensive metrics.

Usage:
    python training/evaluate.py --model checkpoints/best_model.pth --test_csv data/processed/test_split.csv

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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import load_model
from models.loss import get_loss_function
from training.train import SignLanguageDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Tuple of (metrics_dict, predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch in progress_bar:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            logits = model(features, seq_lengths)
            probs = torch.softmax(logits, dim=1)
            _, predictions = torch.max(logits, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision_macro': float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
    }
    
    return metrics, all_preds, all_labels, all_probs


def generate_per_class_metrics(predictions: np.ndarray,
                               labels: np.ndarray,
                               class_names: Dict[int, str]) -> pd.DataFrame:
    """
    Generate per-class metrics.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: Mapping of class index to name
    
    Returns:
        DataFrame with per-class metrics
    """
    num_classes = len(class_names)
    
    records = []
    for class_idx in range(num_classes):
        # Get all samples for this class (ground truth)
        mask = labels == class_idx
        if not mask.any():
            continue
        
        # Binarize: 1 if predicted as this class, 0 otherwise
        binary_labels = (labels == class_idx).astype(int)
        binary_preds = (predictions == class_idx).astype(int)
        
        # Calculate metrics for this class
        support = mask.sum()
        class_correct = (predictions[mask] == class_idx).sum()
        accuracy = float(class_correct / support)
        
        precision = precision_score(binary_labels, binary_preds, average='binary', zero_division=0)
        recall = recall_score(binary_labels, binary_preds, average='binary', zero_division=0)
        f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)
        
        records.append({
            'class_idx': class_idx,
            'class_name': class_names.get(class_idx, f'Class_{class_idx}'),
            'support': int(support),
            'accuracy': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
    
    return pd.DataFrame(records)


def plot_confusion_matrix(predictions: np.ndarray,
                         labels: np.ndarray,
                         class_names: Dict[int, str],
                         save_path: str):
    """
    Plot and save confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: Mapping of class index to name
        save_path: Path to save figure
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sign language model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_split.csv',
                       help='Path to test split CSV')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    
    logger.info(f"Evaluating model on {device}")
    
    # Load model
    model, checkpoint = load_model(args.model, device=str(device))
    
    # Load test data
    logger.info("Loading test data...")
    test_dataset = SignLanguageDataset(
        args.test_csv,
        Path(args.test_csv).parent / 'features'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    logger.info("Evaluating...")
    metrics, predictions, labels, probabilities = evaluate_model(model, test_loader, str(device))
    
    # Log metrics
    logger.info("="*50)
    logger.info("TEST METRICS")
    logger.info("="*50)
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    logger.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Per-class metrics
    class_names = checkpoint.get('class_mapping', {class_idx: f'Class_{class_idx}' for class_idx in range(40)})
    per_class_df = generate_per_class_metrics(predictions, labels, class_names)
    
    per_class_path = output_dir / 'per_class_metrics.csv'
    per_class_df.to_csv(per_class_path, index=False)
    logger.info(f"Per-class metrics saved to {per_class_path}")
    
    # Confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(predictions, labels, class_names, str(cm_path))
    
    # Classification report
    report_path = output_dir / 'classification_report.txt'
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(np.concatenate([labels, predictions])))
    target_names = [class_names.get(i, f'Class_{i}') for i in unique_classes]
    with open(report_path, 'w') as f:
        f.write(classification_report(labels, predictions, labels=unique_classes, 
                                     target_names=target_names, zero_division=0))
    logger.info(f"Classification report saved to {report_path}")
    
    logger.info("="*50)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
