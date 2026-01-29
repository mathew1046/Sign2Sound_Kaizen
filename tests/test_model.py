"""
Test Suite for Model Architecture

Tests for BiLSTM model, loss functions, and forward passes.

Run: pytest tests/test_model.py -v

Author: Team Kaizen
Date: January 2026
"""

import pytest
import torch
import torch.nn as nn

from models.model import BiLSTMClassifier, LSTMClassifier, count_parameters, load_model
from models.loss import WeightedCrossEntropyLoss, FocalLoss, calculate_class_weights


class TestBiLSTMModel:
    """Test BiLSTM classifier."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return BiLSTMClassifier(
            input_size=126,
            hidden_size=256,
            num_layers=2,
            num_classes=40,
            dropout=0.3
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        batch_size = 32
        seq_length = 60
        features = torch.randn(batch_size, seq_length, 126)
        seq_lengths = torch.full((batch_size,), seq_length)
        return features, seq_lengths
    
    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass."""
        features, seq_lengths = sample_input
        output = model(features, seq_lengths)
        
        assert output.shape == (32, 40)  # batch_size x num_classes
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    
    def test_output_range(self, model, sample_input):
        """Test that output logits are in reasonable range."""
        features, seq_lengths = sample_input
        output = model(features, seq_lengths)
        
        assert output.min() > -100, "Output logits too negative"
        assert output.max() < 100, "Output logits too positive"
    
    def test_gradient_flow(self, model, sample_input):
        \"\"\"Test that gradients flow through model.\"\""\n        features, seq_lengths = sample_input
        output = model(features, seq_lengths)
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter has no gradient"
    
    def test_eval_mode(self, model, sample_input):
        """Test model in eval mode."""
        model.eval()
        features, seq_lengths = sample_input
        
        with torch.no_grad():
            output = model(features, seq_lengths)
        
        assert output.shape == (32, 40)
    
    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        seq_length = 60
        
        for batch_size in [1, 16, 64]:
            features = torch.randn(batch_size, seq_length, 126)
            seq_lengths = torch.full((batch_size,), seq_length)
            
            output = model(features, seq_lengths)
            assert output.shape == (batch_size, 40)
    
    def test_different_sequence_lengths(self, model):
        """Test model with different sequence lengths."""
        batch_size = 32
        
        for seq_length in [10, 30, 60, 100]:
            features = torch.randn(batch_size, seq_length, 126)
            seq_lengths = torch.full((batch_size,), seq_length)
            
            output = model(features, seq_lengths)
            assert output.shape == (batch_size, 40)


class TestLSTMModel:
    """Test unidirectional LSTM for comparison."""
    
    @pytest.fixture
    def model(self):
        """Create LSTM model."""
        return LSTMClassifier(
            input_size=126,
            hidden_size=256,
            num_layers=2,
            num_classes=40,
            dropout=0.3
        )
    
    def test_lstm_forward_pass(self, model):
        """Test LSTM forward pass."""
        batch_size = 32
        seq_length = 60
        features = torch.randn(batch_size, seq_length, 126)
        seq_lengths = torch.full((batch_size,), seq_length)
        
        output = model(features, seq_lengths)
        assert output.shape == (batch_size, 40)


class TestLossFunctions:
    """Test loss functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for loss testing."""
        logits = torch.randn(32, 40)  # 32 batch size, 40 classes
        targets = torch.randint(0, 40, (32,))
        class_weights = torch.ones(40)
        class_weights[7:15] = 10.0  # Weight for Malayalam Dynamic
        return logits, targets, class_weights
    
    def test_weighted_cross_entropy(self, sample_data):
        """Test weighted cross-entropy loss."""
        logits, targets, class_weights = sample_data
        loss_fn = WeightedCrossEntropyLoss(class_weights)
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss contains NaN or Inf"
    
    def test_focal_loss(self, sample_data):
        \"\"\"Test focal loss.\"\""\n        logits, targets, class_weights = sample_data
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss contains NaN or Inf"
    
    def test_loss_backward(self, sample_data):
        \"\"\"Test that loss can backpropagate.\"\""\n        logits = sample_data[0].clone().detach().requires_grad_(True)\n        targets, class_weights = sample_data[1], sample_data[2]\n        \n        loss_fn = WeightedCrossEntropyLoss(class_weights)\n        loss = loss_fn(logits, targets)\n        loss.backward()\n        \n        assert logits.grad is not None, "Logits have no gradient"


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_count_parameters(self):
        \"\"\"Test parameter counting.\"\""\n        model = BiLSTMClassifier(\n            input_size=126,\n            hidden_size=256,\n            num_layers=2,\n            num_classes=40,\n            dropout=0.3\n        )\n        \n        param_count = count_parameters(model)\n        \n        assert param_count > 0, "No parameters counted"\n        assert param_count == sum(p.numel() for p in model.parameters()), "Parameter count mismatch"\n    
    def test_calculate_class_weights(self):
        \"\"\"Test class weight calculation.\"\""\n        class_counts = {0: 100, 1: 50, 2: 200}\n        \n        weights = calculate_class_weights(class_counts)\n        \n        assert len(weights) == 3\n        assert all(w > 0 for w in weights), "Weights should be positive"\n        assert weights[0] < weights[1], "Rarer classes should have higher weights"


class TestModelIOD:
    """Test model input/output dimensions."""
    
    def test_input_output_dimensions(self):
        \"\"\"Test model with various input dimensions.\"\""\n        model = BiLSTMClassifier(\n            input_size=126,\n            hidden_size=256,\n            num_layers=2,\n            num_classes=40,\n            dropout=0.3\n        )\n        \n        batch_size = 16\n        seq_len = 60\n        \n        features = torch.randn(batch_size, seq_len, 126)\n        seq_lengths = torch.full((batch_size,), seq_len)\n        \n        output = model(features, seq_lengths)\n        \n        assert output.shape[0] == batch_size, "Batch size mismatch"\n        assert output.shape[1] == 40, "Class dimension mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
