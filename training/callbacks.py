"""
Training Callbacks for BiLSTM Model

This module implements callbacks for:
- Early stopping
- Model checkpointing
- Learning rate scheduling monitoring

Author: Team Kaizen
Date: January 2026
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print updates
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop_flag = False
    
    def update(self, current_loss: float):
        """
        Update early stopping state.
        
        Args:
            current_loss: Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"Early stopping: Loss improved to {current_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping: No improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.should_stop_flag = True
    
    def should_stop(self) -> bool:
        """Return whether training should stop."""
        return self.should_stop_flag
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop_flag = False


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', verbose: bool = True):
        """
        Initialize model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            verbose: Print updates
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def update(self, epoch: int, model: nn.Module, optimizer, 
               train_loss: float, train_acc: float,
               val_loss: float, val_acc: float,
               config: dict = None, model_type: str = 'bilstm', loss_type: str = None):
        """
        Update checkpoint state and save if improved.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            config: Model configuration
            model_type: Type of model ('bilstm' or 'bilstm_ctc')
            loss_type: Loss type used ('weighted_ce', 'ctc', etc)
        """
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = self.checkpoint_dir / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
                'model_type': model_type,
                'loss_type': loss_type
            }, checkpoint_path)
            
            if self.verbose:
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
                'model_type': model_type,
                'loss_type': loss_type
            }, best_path)
            
            if self.verbose:
                logger.info(f"Best model saved: {best_path} (Acc: {val_acc:.2f}%)")


class LearningRateMonitor:
    """Monitor learning rate during training."""
    
    def __init__(self, verbose: bool = True):
        """Initialize learning rate monitor."""
        self.verbose = verbose
        self.lrs = []
    
    def update(self, optimizer):
        """
        Record current learning rate.
        
        Args:
            optimizer: PyTorch optimizer
        """
        current_lr = optimizer.param_groups[0]['lr']
        self.lrs.append(current_lr)
        
        if self.verbose:
            logger.info(f"Current learning rate: {current_lr:.6f}")
    
    def get_lrs(self):
        """Return learning rate history."""
        return self.lrs
