"""
Loss Functions for Sign Language Recognition

This module implements weighted loss functions to handle class imbalance
between Malayalam (classes 0-14) and ISL (classes 15-39).

Author: Team Kaizen
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance.
    
    Weights are calculated as: total_samples / (num_classes * class_count)
    This gives higher weight to under-represented classes.
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted cross-entropy loss.
        
        Args:
            weights: Class weights tensor of shape (num_classes,)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
        Args:
            outputs: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        return self.criterion(outputs, targets)


def calculate_class_weights(train_csv_path: str, 
                           num_classes: int = 40,
                           device: str = 'cpu') -> torch.Tensor:
    """
    Calculate class weights from training data distribution.
    
    Args:
        train_csv_path: Path to train_split.csv
        num_classes: Total number of classes
        device: Device to place tensor on
    
    Returns:
        Class weights tensor of shape (num_classes,)
    """
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    
    # Count samples per class
    class_counts = train_df['class_idx'].value_counts().sort_index()
    
    # Ensure all classes are present
    counts = np.zeros(num_classes)
    for idx, count in class_counts.items():
        if idx < num_classes:
            counts[idx] = count
    
    # Calculate weights: total / (num_classes * count)
    total_samples = counts.sum()
    weights = total_samples / (num_classes * (counts + 1e-10))  # Add epsilon to avoid division by zero
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Log weights
    logger.info("Class weights calculated:")
    for idx in range(min(10, num_classes)):
        logger.info(f"  Class {idx}: {weights[idx]:.4f} (count: {int(counts[idx])})")
    if num_classes > 10:
        logger.info(f"  ... ({num_classes - 10} more classes)")
    
    return weights_tensor


def calculate_class_weights_from_dict(class_distribution: Dict[int, int],
                                     num_classes: int = 40,
                                     device: str = 'cpu') -> torch.Tensor:
    """
    Calculate class weights from class distribution dictionary.
    
    Args:
        class_distribution: Dictionary mapping class_idx -> sample_count
        num_classes: Total number of classes
        device: Device to place tensor on
    
    Returns:
        Class weights tensor of shape (num_classes,)
    """
    counts = np.zeros(num_classes)
    for class_idx, count in class_distribution.items():
        if class_idx < num_classes:
            counts[class_idx] = count
    
    total_samples = counts.sum()
    weights = total_samples / (num_classes * (counts + 1e-10))
    weights = weights / weights.sum() * num_classes
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    return weights_tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean' or 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            outputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.
    Prevents overconfidence by smoothing hard labels.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (default: 0.1)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            outputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        return loss.mean()


class CTCLoss(nn.Module):
    """
    Connectionist Temporal Classification (CTC) Loss.
    
    CTC is suitable for sequence-to-sequence tasks where alignment between
    input and output sequences is unknown. It automatically handles variable
    length sequences and learns the alignment.
    
    Reference: Graves et al. "Connectionist Temporal Classification: Labelling 
    Unsegmented Sequence Data with Recurrent Neural Networks"
    """
    
    def __init__(self, blank: int = 0, reduction: str = 'mean'):
        """
        Initialize CTC loss.
        
        Args:
            blank: Class index to use as blank label (default: 0)
            reduction: Reduction method ('mean' or 'sum')
        """
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(blank=blank, reduction=reduction)
        self.blank = blank
    
    def forward(self, 
                outputs: torch.Tensor, 
                targets: torch.Tensor,
                output_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            outputs: Model outputs of shape (T, batch_size, num_classes)
                    where T is sequence length
            targets: Target sequences of shape (batch_size, target_length)
            output_lengths: Length of output sequences (batch_size,)
            target_lengths: Length of target sequences (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Ensure output_lengths and target_lengths are on CPU
        output_lengths = output_lengths.cpu() if output_lengths.is_cuda else output_lengths
        target_lengths = target_lengths.cpu() if target_lengths.is_cuda else target_lengths
        
        # Flatten targets to 1D for CTC loss
        targets_flat = targets.view(-1)
        
        loss = self.criterion(outputs, targets_flat, output_lengths, target_lengths)
        return loss


def get_loss_function(loss_type: str = 'weighted_ce',
                     class_weights: Optional[torch.Tensor] = None,
                     **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('weighted_ce', 'focal', 'label_smoothing', 'ce', 'ctc')
        class_weights: Class weights for weighted losses
        **kwargs: Additional arguments for specific losses
    
    Returns:
        Loss function module
    """
    if loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(weights=class_weights)
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif loss_type == 'label_smoothing':
        num_classes = kwargs.get('num_classes', 40)
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'ctc':
        blank = kwargs.get('blank', 0)
        return CTCLoss(blank=blank)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 16
    num_classes = 40
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create class weights (simulate imbalance)
    class_dist = {i: 1000 if i >= 15 else 100 for i in range(num_classes)}
    weights = calculate_class_weights_from_dict(class_dist, num_classes)
    
    print(f"✓ Class weights created (min: {weights.min():.4f}, max: {weights.max():.4f})")
    
    # Test weighted CE
    wce_loss = WeightedCrossEntropyLoss(weights=weights)
    loss1 = wce_loss(outputs, targets)
    print(f"✓ Weighted CE Loss: {loss1.item():.4f}")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    loss2 = focal_loss(outputs, targets)
    print(f"✓ Focal Loss: {loss2.item():.4f}")
    
    # Test label smoothing
    ls_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    loss3 = ls_loss(outputs, targets)
    print(f"✓ Label Smoothing Loss: {loss3.item():.4f}")
    
    # Test regular CE
    ce_loss = nn.CrossEntropyLoss()
    loss4 = ce_loss(outputs, targets)
    print(f"✓ Cross-Entropy Loss: {loss4.item():.4f}")
    
    print("\n✓ All tests passed!")
