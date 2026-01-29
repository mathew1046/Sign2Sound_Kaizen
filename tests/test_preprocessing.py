"""
Test Suite for Data Preprocessing

Tests for preprocessing pipeline including feature extraction, augmentation, and splits.

Run: pytest tests/test_preprocessing.py -v

Author: Team Kaizen
Date: January 2026
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import torch

from preprocessing.augmentation import (
    add_gaussian_noise, scale_variation, translate, rotate,
    horizontal_flip, temporal_speed, augment_sample
)
from features.feature_utils import (
    pad_or_truncate, normalize_features, validate_feature_shape,
    create_attention_mask
)


class TestAugmentation:
    """Test data augmentation functions."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return np.random.randn(60, 126).astype(np.float32)
    
    def test_gaussian_noise(self, sample_features):
        """Test Gaussian noise augmentation."""
        augmented = add_gaussian_noise(sample_features, std=0.01)
        
        assert augmented.shape == sample_features.shape
        assert not np.array_equal(augmented, sample_features)
        assert np.allclose(augmented.mean(), sample_features.mean(), atol=0.1)
    
    def test_scale_variation(self, sample_features):
        """Test scale variation augmentation."""
        augmented = scale_variation(sample_features, scale_range=(0.85, 1.15))
        
        assert augmented.shape == sample_features.shape
        assert not np.array_equal(augmented, sample_features)
    
    def test_translate(self, sample_features):
        """Test translation augmentation."""
        augmented = translate(sample_features, translate_range=0.1)
        
        assert augmented.shape == sample_features.shape
        assert not np.array_equal(augmented, sample_features)
    
    def test_rotate(self, sample_features):
        """Test rotation augmentation."""
        augmented = rotate(sample_features, angle_range=15)
        
        assert augmented.shape == sample_features.shape
        assert not np.array_equal(augmented, sample_features)
    
    def test_horizontal_flip(self, sample_features):
        """Test horizontal flip augmentation."""
        augmented = horizontal_flip(sample_features)
        
        assert augmented.shape == sample_features.shape
        # Check that left/right hands are swapped
        left_hand = sample_features[:, :63]
        right_hand = sample_features[:, 63:]
        
        assert augmented.shape == sample_features.shape
    
    def test_temporal_speed(self, sample_features):
        """Test temporal speed augmentation."""
        augmented = temporal_speed(sample_features, speed_range=(0.8, 1.2))
        
        # Shape may change due to interpolation
        assert augmented.ndim == 2
        assert augmented.shape[1] == sample_features.shape[1]
    
    def test_augment_sample(self, sample_features):
        """Test combined augmentation."""
        augmented = augment_sample(sample_features, num_augments=5)
        
        assert len(augmented) == 6  # Original + 5 augmented
        assert all(isinstance(aug, np.ndarray) for aug in augmented)
        assert all(aug.shape == sample_features.shape for aug in augmented[:1])  # Original shape


class TestFeatureUtils:
    """Test feature utility functions."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return np.random.randn(80, 126).astype(np.float32)
    
    def test_pad_or_truncate_padding(self):
        """Test padding of short sequences."""
        features = np.random.randn(40, 126).astype(np.float32)
        padded = pad_or_truncate(features, max_len=60)
        
        assert padded.shape == (60, 126)
        # Check that original values are preserved
        assert np.allclose(padded[:40], features)
        # Check that padding is zeros
        assert np.allclose(padded[40:], 0)
    
    def test_pad_or_truncate_truncation(self):
        \"\"\"Test truncation of long sequences.\"\"\"\n        features = np.random.randn(100, 126).astype(np.float32)
        truncated = pad_or_truncate(features, max_len=60)
        
        assert truncated.shape == (60, 126)
        assert np.allclose(truncated, features[:60])
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.random.randn(60, 126).astype(np.float32) * 10
        normalized = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_validate_feature_shape(self):
        """Test feature shape validation."""
        valid_features = np.random.randn(60, 126).astype(np.float32)
        invalid_features = np.random.randn(60, 100).astype(np.float32)
        
        assert validate_feature_shape(valid_features)
        assert not validate_feature_shape(invalid_features)
    
    def test_create_attention_mask(self):
        """Test attention mask creation."""
        lengths = torch.tensor([30, 60, 45])
        max_len = 60
        
        mask = create_attention_mask(lengths, max_len)
        
        assert mask.shape == (3, max_len)
        # Check first sequence
        assert mask[0, :30].sum() == 30
        assert mask[0, 30:].sum() == 0


class TestDataIntegrity:
    """Test data integrity."""
    
    def test_no_nan_in_augmented(self):
        """Test that augmentation doesn't produce NaN values."""
        features = np.random.randn(60, 126).astype(np.float32)
        
        augmented = augment_sample(features, num_augments=10)
        
        for aug in augmented:
            assert not np.isnan(aug).any(), "Augmentation produced NaN values"
    
    def test_no_inf_in_augmented(self):
        """Test that augmentation doesn't produce Inf values."""
        features = np.random.randn(60, 126).astype(np.float32)
        
        augmented = augment_sample(features, num_augments=10)
        
        for aug in augmented:
            assert not np.isinf(aug).any(), "Augmentation produced Inf values"
    
    def test_feature_range(self):
        """Test that features are in reasonable range."""
        features = np.random.randn(60, 126).astype(np.float32)
        
        assert features.min() >= -5, "Features have very low values"
        assert features.max() <= 5, "Features have very high values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
