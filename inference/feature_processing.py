"""
Enhanced Feature Utilities with Wrist-Relative Normalization

This module provides utility functions for processing hand landmarks with
proper normalization matching the training preprocessing pipeline.

Author: Team Kaizen
Date: January 2026
"""

import numpy as np
from typing import Optional
from collections import deque


def normalize_landmarks_wrist_relative(features: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be relative to wrist position (landmark 0).
    This makes features invariant to hand position in the frame.
    
    Args:
        features: Feature vector of shape (126,) or (60, 126)
                 126 dims = 2 hands × 21 landmarks × 3 coords (x, y, z)
    
    Returns:
        Normalized features with same shape, coordinates relative to wrist
    """
    if features.ndim == 1:
        # Single frame: (126,)
        features_copy = features.copy()
        
        # Process each hand separately
        for hand_idx in range(2):
            start_idx = hand_idx * 63  # 21 landmarks × 3 coords
            end_idx = start_idx + 63
            
            hand_features = features_copy[start_idx:end_idx]
            
            # Check if hand has data (not all zeros)
            if np.any(hand_features):
                # Reshape to (21, 3) for easier manipulation
                landmarks = hand_features.reshape(21, 3)
                
                # Get wrist position (landmark 0)
                wrist = landmarks[0].copy()
                
                # Subtract wrist from all landmarks
                landmarks = landmarks - wrist
                
                # Flatten back and store
                features_copy[start_idx:end_idx] = landmarks.flatten()
        
        return features_copy
    
    elif features.ndim == 2:
        # Sequence: (seq_len, 126)
        normalized_sequence = np.zeros_like(features)
        
        for frame_idx in range(features.shape[0]):
            normalized_sequence[frame_idx] = normalize_landmarks_wrist_relative(features[frame_idx])
        
        return normalized_sequence
    
    else:
        raise ValueError(f"Expected features of shape (126,) or (seq_len, 126), got {features.shape}")


class OneEuroFilter:
    """
    One Euro Filter for smooth tracking with low latency.
    
    Reference: Casiez, G., Roussel, N., & Vogel, D. (2012). 
    1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    """
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """
        Initialize One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a value.
        
        Args:
            x: Current value
            t: Current timestamp (optional, uses counter if None)
        
        Returns:
            Filtered value
        """
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t if t is not None else 0
            return x
        
        # Time delta
        if t is None:
            t = self.t_prev + 1
        dt = t - self.t_prev
        
        if dt <= 0:
            return self.x_prev
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Filter derivative
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_filtered = self._exponential_smoothing(alpha_d, dx, self.dx_prev)
        
        # Compute cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        
        # Filter value
        alpha = self._smoothing_factor(dt, cutoff)
        x_filtered = self._exponential_smoothing(alpha, x, self.x_prev)
        
        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered
        self.t_prev = t
        
        return x_filtered
    
    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        """Calculate smoothing factor (alpha)."""
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)
    
    @staticmethod
    def _exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return alpha * x + (1 - alpha) * x_prev
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class LandmarkSmoother:
    """
    Smooth hand landmarks using One Euro Filter for each coordinate.
    Reduces MediaPipe jitter in real-time tracking.
    """
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        """
        Initialize landmark smoother.
        
        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        
        # Create filter for each coordinate (126 dimensions)
        self.filters = [OneEuroFilter(min_cutoff, beta) for _ in range(126)]
    
    def smooth(self, features: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Smooth feature vector.
        
        Args:
            features: Feature vector of shape (126,)
            timestamp: Optional timestamp
        
        Returns:
            Smoothed features of shape (126,)
        """
        if features.shape != (126,):
            raise ValueError(f"Expected features of shape (126,), got {features.shape}")
        
        smoothed = np.zeros(126, dtype=np.float32)
        
        for i in range(126):
            smoothed[i] = self.filters[i](features[i], timestamp)
        
        return smoothed
    
    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()


class SlidingWindowBuffer:
    """
    Sliding window buffer for sequence-based inference.
    Maintains fixed-length buffer of recent frames using deque.
    """
    
    def __init__(self, window_size: int = 30, feature_dim: int = 126):
        """
        Initialize sliding window buffer.
        
        Args:
            window_size: Number of frames to maintain
            feature_dim: Feature vector dimension
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)
    
    def add(self, features: np.ndarray):
        """
        Add new frame to buffer.
        
        Args:
            features: Feature vector of shape (126,)
        """
        if features.shape != (self.feature_dim,):
            raise ValueError(f"Expected features of shape ({self.feature_dim},), got {features.shape}")
        
        self.buffer.append(features.copy())
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence from buffer.
        
        Returns:
            Array of shape (window_size, feature_dim) if buffer is full,
            None otherwise
        """
        if len(self.buffer) < self.window_size:
            return None
        
        return np.array(list(self.buffer), dtype=np.float32)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.buffer) >= self.window_size
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
    
    def __len__(self):
        """Get current buffer size."""
        return len(self.buffer)


class MovingAverageFilter:
    """
    Simple moving average filter for smoothing.
    Alternative to One Euro Filter for less complex scenarios.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Number of values to average
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def add(self, value: float) -> float:
        """
        Add value and get filtered result.
        
        Args:
            value: New value
        
        Returns:
            Moving average
        """
        self.values.append(value)
        return np.mean(list(self.values))
    
    def reset(self):
        """Reset filter."""
        self.values.clear()


def validate_features(features: np.ndarray) -> bool:
    """
    Validate feature vector for quality.
    
    Args:
        features: Feature vector to validate
    
    Returns:
        True if valid, False otherwise
    """
    if features is None:
        return False
    
    # Check for NaN
    if np.isnan(features).any():
        return False
    
    # Check for Inf
    if np.isinf(features).any():
        return False
    
    # Check if completely zero (no hands detected)
    if not np.any(features):
        return False
    
    return True


def pad_or_truncate(sequence: np.ndarray, max_len: int) -> np.ndarray:
    """
    Pad or truncate sequence to fixed length.
    
    Args:
        sequence: Input sequence of shape (seq_len, feature_dim) or (feature_dim,)
        max_len: Target sequence length
    
    Returns:
        Padded/truncated sequence of shape (max_len, feature_dim)
    """
    if sequence.ndim == 1:
        # Single frame: expand to sequence
        padded = np.zeros((max_len, sequence.shape[0]), dtype=np.float32)
        padded[0] = sequence
        return padded
    
    seq_len, feature_dim = sequence.shape
    
    if seq_len >= max_len:
        # Truncate
        return sequence[:max_len]
    else:
        # Pad with zeros
        padded = np.zeros((max_len, feature_dim), dtype=np.float32)
        padded[:seq_len] = sequence
        return padded
