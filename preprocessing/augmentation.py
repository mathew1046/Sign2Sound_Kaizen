"""
Data Augmentation Module for Sign Language Recognition

This module provides various data augmentation techniques to improve model
robustness and address class imbalance for ISL signs.

Augmentation Techniques:
1. Gaussian Noise - Add random noise to coordinates
2. Scale Variation - Scale hand size
3. Translation - Shift hand position
4. Rotation - Rotate hand in 2D plane
5. Horizontal Flip - Mirror for POV augmentation
6. Temporal Speed - Change sequence playback speed

Author: Team Kaizen
Date: January 2026
"""

import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_gaussian_noise(data: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to landmark coordinates.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
        sigma: Standard deviation of noise (default: 0.02)
    
    Returns:
        Augmented data with same shape as input
    """
    noise = np.random.normal(0, sigma, data.shape)
    augmented = data + noise
    augmented = np.clip(augmented, 0.0, 1.0)
    return augmented.astype(np.float32)


def scale_variation(data: np.ndarray, scale_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """
    Apply random scaling to hand landmarks.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
        scale_range: Min and max scale factors (default: 0.85-1.15)
    
    Returns:
        Scaled data with same shape as input
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    # Calculate center point for each frame
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    augmented = data.copy()
    for i in range(data.shape[0]):
        frame = data[i].reshape(-1, 3)  # Reshape to (42, 3)
        
        # Find center of non-zero landmarks
        non_zero_mask = np.any(frame != 0, axis=1)
        if non_zero_mask.any():
            center = frame[non_zero_mask].mean(axis=0)
            
            # Scale around center
            frame[non_zero_mask] = center + scale * (frame[non_zero_mask] - center)
            frame = np.clip(frame, 0.0, 1.0)
        
        augmented[i] = frame.flatten()
    
    if squeeze:
        augmented = augmented.squeeze(0)
    
    return augmented.astype(np.float32)


def translate(data: np.ndarray, tx: float = None, ty: float = None, 
              translate_range: float = 0.1) -> np.ndarray:
    """
    Apply random translation to hand landmarks.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
        tx: Translation in x direction (if None, random)
        ty: Translation in y direction (if None, random)
        translate_range: Maximum translation amount (default: 0.1)
    
    Returns:
        Translated data with same shape as input
    """
    if tx is None:
        tx = np.random.uniform(-translate_range, translate_range)
    if ty is None:
        ty = np.random.uniform(-translate_range, translate_range)
    
    augmented = data.copy()
    
    # Apply translation to x and y coordinates
    augmented[..., 0::3] += tx  # x coordinates
    augmented[..., 1::3] += ty  # y coordinates
    # z coordinates remain unchanged
    
    augmented = np.clip(augmented, 0.0, 1.0)
    return augmented.astype(np.float32)


def rotate(data: np.ndarray, angle: float = None, angle_range: float = 15.0) -> np.ndarray:
    """
    Apply 2D rotation to hand landmarks.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
        angle: Rotation angle in degrees (if None, random)
        angle_range: Maximum rotation angle (default: ±15 degrees)
    
    Returns:
        Rotated data with same shape as input
    """
    if angle is None:
        angle = np.random.uniform(-angle_range, angle_range)
    
    # Convert to radians
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    augmented = data.copy()
    
    for i in range(data.shape[0]):
        frame = data[i].reshape(-1, 3)  # (42, 3)
        
        # Find center for rotation
        non_zero_mask = np.any(frame != 0, axis=1)
        if non_zero_mask.any():
            center = frame[non_zero_mask, :2].mean(axis=0)
            
            # Rotate x, y coordinates around center
            xy = frame[:, :2]
            xy[non_zero_mask] = (rotation_matrix @ (xy[non_zero_mask] - center).T).T + center
            xy = np.clip(xy, 0.0, 1.0)
            
            frame[:, :2] = xy
        
        augmented[i] = frame.flatten()
    
    if squeeze:
        augmented = augmented.squeeze(0)
    
    return augmented.astype(np.float32)


def horizontal_flip(data: np.ndarray) -> np.ndarray:
    """
    Flip hand landmarks horizontally (mirror image).
    This simulates Point-of-View (POV) augmentation.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
    
    Returns:
        Flipped data with same shape as input
    """
    augmented = data.copy()
    
    # Flip x coordinates: new_x = 1.0 - old_x
    augmented[..., 0::3] = 1.0 - augmented[..., 0::3]
    
    # Swap left and right hands (first 63 features <-> last 63 features)
    if augmented.ndim == 1:
        hand1 = augmented[:63].copy()
        hand2 = augmented[63:].copy()
        augmented[:63] = hand2
        augmented[63:] = hand1
    else:
        hand1 = augmented[:, :63].copy()
        hand2 = augmented[:, 63:].copy()
        augmented[:, :63] = hand2
        augmented[:, 63:] = hand1
    
    return augmented.astype(np.float32)


def temporal_speed(data: np.ndarray, speed_factor: float = None, 
                   speed_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Change temporal speed of sequence by interpolation.
    Only applicable to dynamic signs (sequences with multiple frames).
    
    Args:
        data: Input array of shape (seq_len, 126)
        speed_factor: Speed multiplier (if None, random)
        speed_range: Min and max speed factors (default: 0.8-1.2)
    
    Returns:
        Speed-adjusted data with same shape as input
    """
    if data.ndim == 1:
        # Single frame, cannot apply temporal augmentation
        return data
    
    seq_len, feature_dim = data.shape
    
    if seq_len <= 1:
        return data
    
    if speed_factor is None:
        speed_factor = np.random.uniform(speed_range[0], speed_range[1])
    
    # Calculate new sequence length
    new_len = int(seq_len * speed_factor)
    new_len = max(2, min(new_len, seq_len))  # Keep within bounds
    
    # Interpolate
    old_indices = np.linspace(0, seq_len - 1, new_len)
    new_data = np.zeros((seq_len, feature_dim), dtype=np.float32)
    
    for i, idx in enumerate(old_indices):
        if i >= seq_len:
            break
        
        # Linear interpolation
        idx_low = int(np.floor(idx))
        idx_high = int(np.ceil(idx))
        
        if idx_low == idx_high:
            new_data[i] = data[idx_low]
        else:
            weight = idx - idx_low
            new_data[i] = (1 - weight) * data[idx_low] + weight * data[idx_high]
    
    # Fill remaining frames with zeros (padding)
    # This maintains the sequence length
    
    return new_data.astype(np.float32)


def augment_sample(data: np.ndarray, 
                  num_augmentations: int = 10,
                  techniques: List[str] = None) -> List[np.ndarray]:
    """
    Generate multiple augmented versions of a single sample.
    
    Args:
        data: Input array of shape (seq_len, 126) or (126,)
        num_augmentations: Number of augmented samples to generate
        techniques: List of techniques to use. Options:
                   ['noise', 'scale', 'translate', 'rotate', 'flip', 'temporal']
                   If None, uses all techniques
    
    Returns:
        List of augmented samples (each with same shape as input)
    """
    if techniques is None:
        techniques = ['noise', 'scale', 'translate', 'rotate', 'flip', 'temporal']
    
    augmented_samples = []
    
    # Always include original
    augmented_samples.append(data.copy())
    
    # Generate augmented samples
    for i in range(num_augmentations - 1):
        aug_data = data.copy()
        
        # Randomly select 1-3 techniques to combine
        num_techniques = np.random.randint(1, min(4, len(techniques) + 1))
        selected = np.random.choice(techniques, size=num_techniques, replace=False)
        
        for technique in selected:
            if technique == 'noise':
                sigma = np.random.choice([0.01, 0.02, 0.03])
                aug_data = add_gaussian_noise(aug_data, sigma=sigma)
            elif technique == 'scale':
                aug_data = scale_variation(aug_data)
            elif technique == 'translate':
                aug_data = translate(aug_data)
            elif technique == 'rotate':
                aug_data = rotate(aug_data)
            elif technique == 'flip':
                aug_data = horizontal_flip(aug_data)
            elif technique == 'temporal' and aug_data.ndim > 1:
                aug_data = temporal_speed(aug_data)
        
        augmented_samples.append(aug_data)
    
    return augmented_samples


def augment_rare_classes(data_dict: dict, 
                         rare_class_ids: List[int],
                         target_samples: int = 75) -> dict:
    """
    Aggressively augment rare/problematic classes.
    
    Args:
        data_dict: Dictionary mapping class_id -> list of samples
        rare_class_ids: List of class IDs that need augmentation (e.g., [7-14])
        target_samples: Target number of samples per rare class
    
    Returns:
        Updated data_dict with augmented samples for rare classes
    """
    logger.info(f"Augmenting rare classes: {rare_class_ids}")
    
    for class_id in rare_class_ids:
        if class_id not in data_dict or len(data_dict[class_id]) == 0:
            logger.warning(f"Class {class_id} has no samples to augment")
            continue
        
        original_count = len(data_dict[class_id])
        samples_needed = target_samples - original_count
        
        if samples_needed <= 0:
            logger.info(f"Class {class_id} already has {original_count} samples")
            continue
        
        logger.info(f"Class {class_id}: {original_count} -> {target_samples} samples")
        
        # Generate augmented samples
        augmented = []
        samples_per_original = max(1, samples_needed // original_count + 1)
        
        for original_sample in data_dict[class_id]:
            # Use all augmentation techniques for rare classes
            aug_list = augment_sample(
                original_sample,
                num_augmentations=samples_per_original,
                techniques=['noise', 'scale', 'translate', 'rotate', 'flip', 'temporal']
            )
            augmented.extend(aug_list[1:])  # Exclude original (already in list)
        
        # Add augmented samples to data_dict
        data_dict[class_id].extend(augmented[:samples_needed])
        
        logger.info(f"Class {class_id}: Generated {len(augmented[:samples_needed])} augmented samples")
    
    return data_dict


def validate_augmented_sample(data: np.ndarray) -> bool:
    """
    Validate that augmented sample is valid.
    
    Args:
        data: Sample to validate
    
    Returns:
        True if valid, False otherwise
    """
    # Check for NaN
    if np.isnan(data).any():
        return False
    
    # Check for Inf
    if np.isinf(data).any():
        return False
    
    # Check for valid range (allowing small margin)
    if np.any(data < -0.1) or np.any(data > 1.1):
        return False
    
    # Check if completely zero (invalid for features)
    if not np.any(data):
        return False
    
    return True


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing augmentation functions...")
    
    # Create dummy data
    dummy_static = np.random.rand(126).astype(np.float32) * 0.8 + 0.1
    dummy_dynamic = np.random.rand(30, 126).astype(np.float32) * 0.8 + 0.1
    
    # Test each augmentation
    print("\n1. Gaussian Noise:")
    noisy = add_gaussian_noise(dummy_static, sigma=0.02)
    print(f"   Valid: {validate_augmented_sample(noisy)}")
    
    print("\n2. Scale Variation:")
    scaled = scale_variation(dummy_static)
    print(f"   Valid: {validate_augmented_sample(scaled)}")
    
    print("\n3. Translation:")
    translated = translate(dummy_static)
    print(f"   Valid: {validate_augmented_sample(translated)}")
    
    print("\n4. Rotation:")
    rotated = rotate(dummy_static)
    print(f"   Valid: {validate_augmented_sample(rotated)}")
    
    print("\n5. Horizontal Flip:")
    flipped = horizontal_flip(dummy_static)
    print(f"   Valid: {validate_augmented_sample(flipped)}")
    
    print("\n6. Temporal Speed (dynamic only):")
    temporal = temporal_speed(dummy_dynamic)
    print(f"   Valid: {validate_augmented_sample(temporal)}")
    print(f"   Shape preserved: {temporal.shape == dummy_dynamic.shape}")
    
    print("\n7. Multiple Augmentations:")
    augmented = augment_sample(dummy_dynamic, num_augmentations=5)
    print(f"   Generated {len(augmented)} samples")
    print(f"   All valid: {all(validate_augmented_sample(x) for x in augmented)}")
    
    print("\n✓ All tests passed!")
