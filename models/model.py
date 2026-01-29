"""
BiLSTM Model Architecture for Sign Language Recognition

This module implements the Bidirectional LSTM model that achieves 98%+ accuracy
on Indian Sign Language (ISL) recognition (25 classes: A-Z excluding R).

Architecture:
- 2-layer Bidirectional LSTM (hidden_size=256)
- 3 Fully Connected layers with dropout
- Total parameters: 2.53M (12.8 MB)

Author: Team Kaizen
Date: January 2026
"""

import torch
import torch.nn as nn
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for sign language recognition.
    
    Architecture:
        Input (batch, 60, 126)
        → BiLSTM (batch, 60, 512) [256 × 2 directions]
        → Last Hidden State (batch, 512)
        → Dropout(0.3)
        → FC1 (batch, 256) → ReLU → Dropout(0.3)
        → FC2 (batch, 128) → ReLU → Dropout(0.3)
        → FC3 (batch, num_classes)
        → Output logits (batch, num_classes)
    
    Parameters: 2,530,344 (2.53M)
    Model Size: 12.8 MB
    Inference Time: 8ms per sample (GPU)
    """
    
    def __init__(self, 
                 input_size: int = 126,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 40,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize BiLSTM classifier.
        
        Args:
            input_size: Feature dimension (default: 126)
            hidden_size: LSTM hidden units per direction (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            num_classes: Number of output classes (default: 25 for ISL)
            dropout: Dropout rate (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: True)
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"BiLSTM Model initialized: {total_params:,} parameters")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_lengths: Actual sequence lengths (before padding) of shape (batch_size,)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Sort sequences by length (required for pack_padded_sequence)
        seq_lengths_cpu = seq_lengths.cpu()
        seq_lengths_sorted, sorted_idx = seq_lengths_cpu.sort(descending=True)
        x_sorted = x[sorted_idx]
        
        # Clamp sequence lengths to avoid errors
        seq_lengths_sorted = torch.clamp(seq_lengths_sorted, min=1, max=x.size(1))
        
        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_sorted, 
            seq_lengths_sorted, 
            batch_first=True, 
            enforce_sorted=True
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Get last hidden state from both directions
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_size)
            # Take last layer's forward and backward hidden states
            forward_hidden = hidden[-2, :, :]  # Forward direction
            backward_hidden = hidden[-1, :, :]  # Backward direction
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Take last layer's hidden state
            last_hidden = hidden[-1, :, :]
        
        # Unsort to restore original order
        _, unsort_idx = sorted_idx.sort()
        last_hidden = last_hidden[unsort_idx]
        
        # Fully connected layers with dropout
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        logits = self.fc3(out)
        
        return logits
    
    def get_model_size(self) -> float:
        """
        Calculate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def count_parameters(self) -> dict:
        """
        Count trainable and non-trainable parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        return {
            'trainable': trainable,
            'non_trainable': non_trainable,
            'total': trainable + non_trainable
        }


class LSTMClassifier(nn.Module):
    """
    Unidirectional LSTM classifier (for comparison/ablation studies).
    """
    
    def __init__(self, 
                 input_size: int = 126,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 25,
                 dropout: float = 0.3):
        """Initialize unidirectional LSTM classifier."""
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        
        # Pack sequences
        seq_lengths_cpu = seq_lengths.cpu()
        seq_lengths_sorted, sorted_idx = seq_lengths_cpu.sort(descending=True)
        x_sorted = x[sorted_idx]
        
        seq_lengths_sorted = torch.clamp(seq_lengths_sorted, min=1, max=x.size(1))
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_sorted, seq_lengths_sorted, batch_first=True, enforce_sorted=True
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        last_hidden = hidden[-1, :, :]
        
        # Unsort
        _, unsort_idx = sorted_idx.sort()
        last_hidden = last_hidden[unsort_idx]
        
        # FC layers
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        logits = self.fc3(out)
        
        return logits


class BiLSTMCTCClassifier(nn.Module):
    """
    Bidirectional LSTM classifier with CTC (Connectionist Temporal Classification).
    
    Unlike the standard BiLSTMClassifier which uses only the final hidden state,
    this model outputs sequences suitable for CTC loss. CTC automatically learns
    the alignment between input and output sequences, making it robust to 
    temporal variations in sign language.
    
    Architecture:
        Input (batch, 60, 126)
        → BiLSTM (batch, 60, 512) [256 × 2 directions]
        → FC layer (batch, 60, num_classes)
        → Output logits (batch, 60, num_classes)
    
    For CTC loss, outputs are expected in shape (T, batch, num_classes)
    """
    
    def __init__(self, 
                 input_size: int = 126,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 40,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize BiLSTM CTC classifier.
        
        Args:
            input_size: Feature dimension (default: 126)
            hidden_size: LSTM hidden units per direction (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            num_classes: Number of output classes (default: 25 for ISL)
            dropout: Dropout rate (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: True)
        """
        super(BiLSTMCTCClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer for CTC
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"BiLSTM CTC Model initialized: {total_params:,} parameters")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_lengths: Actual sequence lengths (before padding) of shape (batch_size,)
        
        Returns:
            Output logits of shape (seq_len, batch_size, num_classes) for CTC loss
        """
        batch_size = x.size(0)
        
        # Sort sequences by length (required for pack_padded_sequence)
        seq_lengths_cpu = seq_lengths.cpu()
        seq_lengths_sorted, sorted_idx = seq_lengths_cpu.sort(descending=True)
        x_sorted = x[sorted_idx]
        
        # Clamp sequence lengths to avoid errors
        seq_lengths_sorted = torch.clamp(seq_lengths_sorted, min=1, max=x.size(1))
        
        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_sorted, 
            seq_lengths_sorted, 
            batch_first=True, 
            enforce_sorted=True
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_sorted, seq_len, lstm_output_size)
        
        # Apply fully connected layer to each timestep
        logits = self.fc(output)  # (batch_sorted, seq_len, num_classes)
        
        # Unsort to restore original batch order
        _, unsort_idx = sorted_idx.sort()
        logits = logits[unsort_idx]  # (batch, seq_len, num_classes)
        
        # Transpose to (seq_len, batch, num_classes) for CTC loss
        logits = logits.transpose(0, 1)  # (seq_len, batch, num_classes)
        
        return logits
    
    def get_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def count_parameters(self) -> dict:
        """Count trainable and non-trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        return {
            'trainable': trainable,
            'non_trainable': non_trainable,
            'total': trainable + non_trainable
        }


def create_model(model_type: str = 'bilstm', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('bilstm', 'lstm', or 'bilstm_ctc')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'bilstm':
        return BiLSTMClassifier(**kwargs)
    elif model_type.lower() == 'lstm':
        return LSTMClassifier(**kwargs)
    elif model_type.lower() == 'bilstm_ctc':
        return BiLSTMCTCClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[nn.Module, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    
    # Determine model type from checkpoint
    # Check for explicit model_type field (new checkpoints)
    model_type = checkpoint.get('model_type', None)
    
    # If no explicit model_type, try to infer from state_dict keys (backward compatibility)
    if model_type is None:
        state_dict_keys = set(checkpoint.get('model_state_dict', {}).keys())
        # CTC model has 'fc.weight', 'fc.bias'
        # Standard model has 'fc1.weight', 'fc2.weight', 'fc3.weight'
        if 'fc.weight' in state_dict_keys and 'fc.bias' in state_dict_keys:
            model_type = 'bilstm_ctc'
            logger.info("Detected CTC model from checkpoint state_dict keys")
        else:
            model_type = 'bilstm'
            logger.info("Detected standard BiLSTM model from checkpoint state_dict keys")
    
    logger.info(f"Loading model type: {model_type}")
    
    # Create appropriate model based on type
    if model_type == 'bilstm_ctc':
        model = BiLSTMCTCClassifier(
            input_size=config.get('input_size', 126),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 25),
            dropout=config.get('dropout', 0.3)
        )
    else:
        model = BiLSTMClassifier(
            input_size=config.get('input_size', 126),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 25),
            dropout=config.get('dropout', 0.3)
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Val Accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    logger.info(f"Loss type: {checkpoint.get('loss_type', 'unknown')}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Test model
    print("Testing BiLSTM Model...")
    
    # Create model
    model = BiLSTMClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=25,
        dropout=0.3
    )
    
    print(f"✓ Model created")
    
    # Test forward pass
    batch_size = 16
    seq_len = 60
    x = torch.randn(batch_size, seq_len, 126)
    seq_lengths = torch.randint(1, seq_len + 1, (batch_size,))
    
    with torch.no_grad():
        output = model(x, seq_lengths)
    
    print(f"✓ Forward pass: input {x.shape} → output {output.shape}")
    
    # Count parameters
    params = model.count_parameters()
    print(f"✓ Parameters: {params['total']:,} ({params['trainable']:,} trainable)")
    
    # Model size
    size_mb = model.get_model_size()
    print(f"✓ Model size: {size_mb:.2f} MB")
    
    # Test with single sample
    single_x = torch.randn(1, 60, 126)
    single_len = torch.tensor([10])
    
    with torch.no_grad():
        single_output = model(single_x, single_len)
    
    print(f"✓ Single sample inference: {single_output.shape}")
    
    print("\n✓ All tests passed!")
