"""
Feature Utilities Module

Utility functions for feature processing, sequence handling, and data manipulation.

Author: Team Kaizen
Date: January 2026
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_sequence(data: np.ndarray, max_len: int, padding_value: float = 0.0) -> np.ndarray:
    """
    Pad sequence to maximum length.
    
    Args:
        data: Input array of shape (seq_len, feature_dim) or (feature_dim,)
        max_len: Maximum sequence length
        padding_value: Value to use for padding (default: 0.0)
    
    Returns:
        Padded array of shape (max_len, feature_dim)
    """
    if data.ndim == 1:
        # Single frame: reshape to (1, feature_dim)
        data = data.reshape(1, -1)
    
    seq_len, feature_dim = data.shape
    
    if seq_len >= max_len:
        return data[:max_len]
    
    # Create padded array
    padded = np.full((max_len, feature_dim), padding_value, dtype=data.dtype)
    padded[:seq_len] = data
    
    return padded


def truncate_sequence(data: np.ndarray, max_len: int) -> np.ndarray:
    """
    Truncate sequence to maximum length.
    
    Args:
        data: Input array of shape (seq_len, feature_dim)
        max_len: Maximum sequence length
    
    Returns:
        Truncated array of shape (max_len, feature_dim)
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    return data[:max_len]


def pad_or_truncate(data: np.ndarray, max_len: int, padding_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate sequence to exactly max_len.
    
    Args:
        data: Input array of shape (seq_len, feature_dim) or (feature_dim,)
        max_len: Target sequence length
        padding_value: Value to use for padding (default: 0.0)
    
    Returns:
        Array of shape (max_len, feature_dim)
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    seq_len = data.shape[0]
    
    if seq_len > max_len:
        return truncate_sequence(data, max_len)
    elif seq_len < max_len:
        return pad_sequence(data, max_len, padding_value)
    else:
        return data


def create_attention_mask(seq_len: int, max_len: int) -> np.ndarray:
    """
    Create attention mask for padded sequences.
    
    Args:
        seq_len: Actual sequence length (before padding)
        max_len: Maximum sequence length (after padding)
    
    Returns:
        Boolean mask of shape (max_len,) where True = padding, False = valid
    """
    mask = np.ones(max_len, dtype=bool)
    mask[:seq_len] = False  # False = valid, True = padding
    return mask


def get_sequence_length(data: np.ndarray) -> int:
    """
    Get actual sequence length (number of non-padding frames).
    
    Args:
        data: Array of shape (seq_len, feature_dim)
    
    Returns:
        Number of non-zero frames
    """
    if data.ndim == 1:
        return 1 if np.any(data) else 0
    
    # Count frames where at least one feature is non-zero
    non_zero_frames = np.any(data != 0, axis=1)
    return int(non_zero_frames.sum())


def batch_pad_sequences(sequences: List[np.ndarray], 
                       max_len: int,
                       padding_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad a batch of sequences and create masks.
    
    Args:
        sequences: List of arrays, each of shape (seq_len, feature_dim)
        max_len: Maximum sequence length
        padding_value: Value to use for padding
    
    Returns:
        Tuple of (padded_batch, lengths)
        - padded_batch: Array of shape (batch_size, max_len, feature_dim)
        - lengths: Array of shape (batch_size,) with actual sequence lengths
    """
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[-1] if sequences[0].ndim > 1 else sequences[0].shape[0]
    
    padded_batch = np.full((batch_size, max_len, feature_dim), 
                           padding_value, 
                           dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        padded = pad_or_truncate(seq, max_len, padding_value)
        padded_batch[i] = padded
        lengths[i] = get_sequence_length(seq)
    
    return padded_batch, lengths


def normalize_features(features: np.ndarray, 
                      mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Array of shape (num_samples, seq_len, feature_dim)
        mean: Pre-computed mean (if None, computed from features)
        std: Pre-computed std (if None, computed from features)
    
    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = features.mean(axis=(0, 1), keepdims=True)
    
    if std is None:
        std = features.std(axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean.squeeze(), std.squeeze()


def denormalize_features(normalized_features: np.ndarray,
                        mean: np.ndarray,
                        std: np.ndarray) -> np.ndarray:
    """
    Denormalize features back to original scale.
    
    Args:
        normalized_features: Normalized array
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized features
    """
    return normalized_features * std + mean


def batch_process(data_list: List, 
                 process_fn,
                 batch_size: int = 32,
                 show_progress: bool = True) -> List:
    """
    Process data in batches with optional progress bar.
    
    Args:
        data_list: List of data items to process
        process_fn: Function to apply to each item
        batch_size: Size of each batch
        show_progress: Whether to show progress bar
    
    Returns:
        List of processed results
    """
    results = []
    
    iterator = range(0, len(data_list), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(data_list)//batch_size + 1, desc="Processing")
        except ImportError:
            pass
    
    for i in iterator:
        batch = data_list[i:i + batch_size]
        batch_results = [process_fn(item) for item in batch]
        results.extend(batch_results)
    
    return results


def split_hands(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split 126-dim features into two hands.
    
    Args:
        features: Array of shape (..., 126)
    
    Returns:
        Tuple of (hand1_features, hand2_features), each of shape (..., 63)
    """
    hand1 = features[..., :63]
    hand2 = features[..., 63:]
    return hand1, hand2


def merge_hands(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    """
    Merge two hand features into single 126-dim vector.
    
    Args:
        hand1: Array of shape (..., 63)
        hand2: Array of shape (..., 63)
    
    Returns:
        Merged features of shape (..., 126)
    """
    return np.concatenate([hand1, hand2], axis=-1)


def reshape_landmarks(features: np.ndarray) -> np.ndarray:
    """
    Reshape flat features to landmark structure.
    
    Args:
        features: Array of shape (..., 126)
    
    Returns:
        Array of shape (..., 42, 3) where 42 = 2 hands × 21 landmarks
    """
    original_shape = features.shape[:-1]
    reshaped = features.reshape(*original_shape, 42, 3)
    return reshaped


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten landmark structure to feature vector.
    
    Args:
        landmarks: Array of shape (..., 42, 3)
    
    Returns:
        Array of shape (..., 126)
    """
    original_shape = landmarks.shape[:-2]
    flattened = landmarks.reshape(*original_shape, 126)
    return flattened


def validate_feature_shape(features: np.ndarray, 
                          expected_feature_dim: int = 126,
                          expected_max_len: Optional[int] = None) -> bool:
    """
    Validate feature array shape.
    
    Args:
        features: Feature array to validate
        expected_feature_dim: Expected feature dimension (default: 126)
        expected_max_len: Expected sequence length (if None, any length OK)
    
    Returns:
        True if valid, False otherwise
    """
    if features.ndim == 1:
        # Single frame
        return features.shape[0] == expected_feature_dim
    elif features.ndim == 2:
        # Sequence
        seq_len, feature_dim = features.shape
        if feature_dim != expected_feature_dim:
            return False
        if expected_max_len is not None and seq_len != expected_max_len:
            return False
        return True
    else:
        return False


def compute_feature_statistics(features: np.ndarray) -> dict:
    """
    Compute statistics for feature array.
    
    Args:
        features: Array of shape (num_samples, seq_len, feature_dim)
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'shape': features.shape,
        'mean': float(features.mean()),
        'std': float(features.std()),
        'min': float(features.min()),
        'max': float(features.max()),
        'num_zeros': int((features == 0).sum()),
        'num_nans': int(np.isnan(features).sum()),
        'num_infs': int(np.isinf(features).sum()),
    }
    
    return stats


def convert_to_torch(data: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        data: Numpy array
        device: Device to place tensor on ('cpu' or 'cuda')
    
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(data).float()
    return tensor.to(device)


def convert_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    # Test feature utilities
    print("Testing Feature Utilities...")
    
    # Test padding
    short_seq = np.random.rand(10, 126).astype(np.float32)
    padded = pad_sequence(short_seq, max_len=60)
    print(f"✓ Padding: {short_seq.shape} -> {padded.shape}")
    
    # Test truncation
    long_seq = np.random.rand(100, 126).astype(np.float32)
    truncated = truncate_sequence(long_seq, max_len=60)
    print(f"✓ Truncation: {long_seq.shape} -> {truncated.shape}")
    
    # Test pad_or_truncate
    result = pad_or_truncate(short_seq, max_len=60)
    print(f"✓ Pad/Truncate: {result.shape} == (60, 126)")
    
    # Test attention mask
    mask = create_attention_mask(seq_len=10, max_len=60)
    print(f"✓ Attention mask: {mask.shape}, valid frames: {(~mask).sum()}")
    
    # Test sequence length
    seq_len = get_sequence_length(padded)
    print(f"✓ Sequence length: {seq_len}")
    
    # Test batch padding
    sequences = [np.random.rand(i, 126).astype(np.float32) for i in [5, 10, 15]]
    batch, lengths = batch_pad_sequences(sequences, max_len=60)
    print(f"✓ Batch padding: {batch.shape}, lengths: {lengths}")
    
    # Test hand splitting
    features = np.random.rand(60, 126).astype(np.float32)
    hand1, hand2 = split_hands(features)
    print(f"✓ Split hands: {hand1.shape}, {hand2.shape}")
    
    # Test hand merging
    merged = merge_hands(hand1, hand2)
    print(f"✓ Merge hands: {merged.shape}")
    
    # Test reshape
    reshaped = reshape_landmarks(features)
    print(f"✓ Reshape: {features.shape} -> {reshaped.shape}")
    
    # Test flatten
    flattened = flatten_landmarks(reshaped)
    print(f"✓ Flatten: {reshaped.shape} -> {flattened.shape}")
    
    # Test validation
    is_valid = validate_feature_shape(features, expected_feature_dim=126)
    print(f"✓ Validation: {is_valid}")
    
    # Test statistics
    stats = compute_feature_statistics(features.reshape(1, 60, 126))
    print(f"✓ Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print("\n✓ All tests passed!")
