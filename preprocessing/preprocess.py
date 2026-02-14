"""
Main Preprocessing Pipeline

This script orchestrates the complete preprocessing pipeline:
1. Load ISL dataset
2. Extract MediaPipe hand landmarks (126-dimensional features)
3. Apply data augmentation (POV flip)
4. Create stratified train/val/test splits (70/15/15)
5. Save processed features and metadata

Usage:
    python preprocessing/preprocess.py \
        --isl_path /path/to/ISL \
        --output data/processed

Author: Team Kaizen
Date: January 2026
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_features import MediaPipeExtractor
from augmentation import horizontal_flip
from inference.feature_processing import normalize_landmarks_wrist_relative

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Class mapping (ISL only, A-Z excluding R)
CLASS_NAMES = {
    0: "ISL_A", 1: "ISL_B", 2: "ISL_C", 3: "ISL_D", 4: "ISL_E",
    5: "ISL_F", 6: "ISL_G", 7: "ISL_H", 8: "ISL_I", 9: "ISL_J",
    10: "ISL_K", 11: "ISL_L", 12: "ISL_M", 13: "ISL_N", 14: "ISL_O",
    15: "ISL_P", 16: "ISL_Q", 17: "ISL_S", 18: "ISL_T", 19: "ISL_U",
    20: "ISL_V", 21: "ISL_W", 22: "ISL_X", 23: "ISL_Y", 24: "ISL_Z"
}


def load_isl_data(isl_path: str) -> Dict[int, List[np.ndarray]]:
    """
    Load ISL dataset (Classes 15-39).
    ISL data is already in .npy format.
    
    Args:
        isl_path: Path to ISL/data directory
    
    Returns:
        Dictionary mapping class_id -> list of feature vectors
    """
    logger.info("Loading ISL data...")
    data_path = Path(isl_path) / "data"
    
    if not data_path.exists():
        logger.error(f"ISL data directory not found: {data_path}")
        return {}
    
    # Map alphabet to class ID (A=0, B=1, ..., Z=24, skip R)
    alphabet = "ABCDEFGHIJKLMNOPQSTUVWXYZ"  # No R
    alphabet_to_class = {letter: idx for idx, letter in enumerate(alphabet)}
    
    data_dict = {}
    
    for letter, class_idx in tqdm(alphabet_to_class.items(), desc="Loading ISL"):
        letter_folder = data_path / letter
        
        if not letter_folder.exists():
            logger.warning(f"Folder not found: {letter_folder}")
            continue
        
        features_list = []
        
        # ISL structure: data/A/0/0.npy, data/A/0/1.npy, ...
        # Iterate through subfolders (0-119)
        for subfolder in sorted(letter_folder.iterdir()):
            if not subfolder.is_dir():
                continue
            
            # Load all .npy files in subfolder
            npy_files = list(subfolder.glob("*.npy"))
            
            for npy_file in npy_files:
                try:
                    features = np.load(str(npy_file))
                    
                    # ISL features should be (126,) shape
                    if features.shape == (126,):
                        # Validate
                        if not np.isnan(features).any() and not np.isinf(features).any() and np.any(features):
                            normalized = normalize_landmarks_wrist_relative(features.astype(np.float32))
                            features_list.append(normalized)
                except Exception as e:
                    logger.warning(f"Failed to load {npy_file}: {e}")
        
        data_dict[class_idx] = features_list
        logger.info(f"Class {class_idx} ({CLASS_NAMES[class_idx]}): Loaded {len(features_list)} samples")
    
    return data_dict


def apply_augmentation(data_dict: Dict[int, List[np.ndarray]], 
                      augment_count: int = 0,
                      max_seq_len: int = 60) -> Dict[int, List[np.ndarray]]:
    """
    Apply augmentation to all classes.
    
    Args:
        data_dict: Dictionary mapping class_id -> list of samples
        augment_count: Unused for ISL-only pipeline (reserved)
        max_seq_len: Maximum sequence length
    
    Returns:
        Augmented data dictionary
    """
    logger.info("Applying data augmentation...")
    
    # POV flip for ISL
    logger.info("Augmenting ISL (POV flip)...")
    for class_idx in sorted(data_dict.keys()):
        if class_idx in data_dict:
            original_samples = data_dict[class_idx].copy()
            flipped_samples = [horizontal_flip(s) for s in original_samples]
            data_dict[class_idx].extend(flipped_samples)
            logger.info(f"Class {class_idx}: {len(original_samples)} -> {len(data_dict[class_idx])} samples")
    
    # Ensure all samples have correct shape
    for class_idx in data_dict:
        formatted_samples = []
        for sample in data_dict[class_idx]:
            if sample.ndim == 1:
                # Static sign: pad to (max_seq_len, 126)
                padded = np.zeros((max_seq_len, 126), dtype=np.float32)
                padded[0] = sample
                formatted_samples.append(padded)
            else:
                # Already a sequence
                formatted_samples.append(sample)
        data_dict[class_idx] = formatted_samples
    
    return data_dict


def create_splits(data_dict: Dict[int, List[np.ndarray]], 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Args:
        data_dict: Dictionary mapping class_id -> list of samples
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Creating train/val/test splits...")
    
    # Create DataFrame with all samples
    records = []
    sample_counter = 0
    
    for class_idx, samples in data_dict.items():
        for sample_idx, sample in enumerate(samples):
            records.append({
                'sample_id': sample_counter,
                'class_idx': class_idx,
                'class_name': CLASS_NAMES[class_idx],
                'sample_data': sample,
                'seq_length': np.count_nonzero(np.any(sample != 0, axis=1))
            })
            sample_counter += 1
    
    df = pd.DataFrame(records)
    logger.info(f"Total samples: {len(df)}")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        stratify=df['class_idx'],
        random_state=random_state
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df['class_idx'],
        random_state=random_state
    )
    
    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_processed_data(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame,
                       output_dir: str):
    """
    Save processed features and metadata.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory path
    """
    logger.info(f"Saving processed data to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    features_dir = output_path / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Save features as .npy files
    def save_split(df, split_name):
        split_records = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split_name}"):
            # Generate filename
            filename = f"class_{row['class_idx']}_sample_{row['sample_id']}.npy"
            filepath = features_dir / filename
            
            # Save features
            np.save(str(filepath), row['sample_data'])
            
            # Record metadata
            split_records.append({
                'sample_id': row['sample_id'],
                'class_idx': row['class_idx'],
                'class_name': row['class_name'],
                'file_path': str(filepath),
                'seq_length': row['seq_length']
            })
        
        # Save CSV
        split_df = pd.DataFrame(split_records)
        csv_path = output_path / f"{split_name}_split.csv"
        split_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {split_name} split: {len(split_df)} samples -> {csv_path}")
        
        return split_df
    
    # Save all splits
    train_meta = save_split(train_df, "train")
    val_meta = save_split(val_df, "val")
    test_meta = save_split(test_df, "test")
    
    # Save class mapping
    class_mapping_df = pd.DataFrame([
        {'class_idx': idx, 'class_name': name}
        for idx, name in CLASS_NAMES.items()
    ])
    class_mapping_path = output_path / "class_mapping.csv"
    class_mapping_df.to_csv(class_mapping_path, index=False)
    logger.info(f"Saved class mapping: {class_mapping_path}")
    
    # Save preprocessing summary
    summary = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'num_classes': len(CLASS_NAMES),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'feature_dim': 126,
        'max_seq_len': 60,
        'class_distribution': {
            class_name: {
                'train': int((train_meta['class_idx'] == idx).sum()),
                'val': int((val_meta['class_idx'] == idx).sum()),
                'test': int((test_meta['class_idx'] == idx).sum())
            }
            for idx, class_name in CLASS_NAMES.items()
        }
    }
    
    summary_path = output_path / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved preprocessing summary: {summary_path}")


def validate_dataset_paths(isl_path: str):
    """
    Validate that dataset paths exist and have correct structure.
    
    Args:
        isl_path: Path to ISL directory
    
    Raises:
        ValueError: If paths are invalid or structure is incorrect
    """
    # Check ISL path
    isl_base = Path(isl_path)
    if not isl_base.exists():
        raise ValueError(f"❌ ISL path does not exist: {isl_path}")
    
    isl_data_path = isl_base / "data"
    if not isl_data_path.exists():
        raise ValueError(f"❌ ISL/data directory not found: {isl_data_path}")
    
    logger.info(f"✓ ISL path validated: {isl_path}")


def validate_loaded_data(all_data: Dict[int, List[np.ndarray]]):
    """
    Validate that all required classes were loaded.
    
    Args:
        all_data: Dictionary mapping class_id -> list of samples
    
    Raises:
        ValueError: If critical data is missing
    """
    isl_count = sum(len(v) for k, v in all_data.items() if 0 <= k < len(CLASS_NAMES))
    
    logger.info("")
    logger.info("="*70)
    logger.info("DATA LOADING SUMMARY")
    logger.info("="*70)
    logger.info(f"ISL (Classes 0-24):               {isl_count:6d} samples")
    logger.info(f"{'─'*70}")
    logger.info(f"Total:                            {isl_count:6d} samples")
    logger.info("="*70)
    
    # Validation
    if isl_count == 0:
        logger.error("❌ ERROR: No ISL data loaded!")
        logger.error("   Expected: 6,000+ samples")
        logger.error("   Actual: 0 samples")
        raise ValueError("ISL data not loaded")
    
    logger.info("✓ All datasets loaded successfully!")
    logger.info("")
    
    return {
        'isl': isl_count,
        'total': isl_count
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess sign language datasets")
    parser.add_argument('--isl_path', type=str, required=True,
                       help='Path to ISL dataset directory')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--augment_count', type=int, default=0,
                       help='Unused for ISL-only pipeline (reserved)')
    parser.add_argument('--max_seq_len', type=int, default=60,
                       help='Maximum sequence length')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='MediaPipe detection confidence threshold')
    
    args = parser.parse_args()
    
    # Validate paths before processing
    try:
        validate_dataset_paths(args.isl_path)
    except ValueError as e:
        logger.error(f"\n{e}\n")
        logger.error("Please ensure:")
        logger.error("  1. ISL_PATH is set correctly")
        logger.error("  2. ISL directory contains the required subdirectories")
        logger.error("")
        logger.error("Example:")
        logger.error("  export ISL_PATH=/path/to/ISL")
        logger.error("  python preprocessing/preprocess.py \\")
        logger.error("    --isl_path \"$ISL_PATH\"")
        exit(1)
    
    logger.info("="*50)
    logger.info("SIGN LANGUAGE PREPROCESSING PIPELINE")
    logger.info("="*50)
    
    # Initialize MediaPipe extractor
    extractor = MediaPipeExtractor(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence
    )
    
    # Load datasets (ISL only)
    isl_data = load_isl_data(args.isl_path)
    
    # Combine all data
    all_data = {**isl_data}
    
    # Validate loaded data
    try:
        data_stats = validate_loaded_data(all_data)
    except ValueError as e:
        logger.error(f"\n{e}\n")
        logger.error("Failed to load required datasets.")
        logger.error("Please verify:")
        logger.error("  1. Dataset paths are correct")
        logger.error("  2. Datasets contain the expected directories")
        logger.error("  3. Datasets are not corrupted or empty")
        exit(1)
    
    # Log statistics
    total_samples = sum(len(samples) for samples in all_data.values())
    logger.info(f"\nLoaded {total_samples} samples from {len(all_data)} classes")
    
    # Apply augmentation
    augmented_data = apply_augmentation(all_data, args.augment_count, args.max_seq_len)
    
    # Log augmented statistics
    total_augmented = sum(len(samples) for samples in augmented_data.values())
    logger.info(f"\nAfter augmentation: {total_augmented} samples")
    
    # Create splits
    train_df, val_df, test_df = create_splits(augmented_data)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, args.output)
    
    # Clean up
    extractor.close()
    
    logger.info("\n" + "="*50)
    logger.info("✓ PREPROCESSING COMPLETE!")
    logger.info("="*50)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Total samples: {total_augmented}")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()
