"""
Main Preprocessing Pipeline

This script orchestrates the complete preprocessing pipeline:
1. Load Malayalam (Static + Dynamic) and ISL datasets
2. Extract MediaPipe hand landmarks (126-dimensional features)
3. Apply data augmentation (aggressive for classes 7-14)
4. Create stratified train/val/test splits (70/15/15)
5. Save processed features and metadata

Usage:
    python preprocessing/preprocess.py \
        --malayalam_path /path/to/MALAYALAM \
        --isl_path /path/to/ISL \
        --output data/processed \
        --augment_count 75

Author: Team Kaizen
Date: January 2026
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from extract_features import MediaPipeExtractor
from augmentation import augment_rare_classes, horizontal_flip, validate_augmented_sample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Class mapping
CLASS_NAMES = {
    # Malayalam Static (0-6)
    0: "Malayalam_അ", 1: "Malayalam_ആ", 2: "Malayalam_ഇ",
    3: "Malayalam_ഉ", 4: "Malayalam_ഋ", 5: "Malayalam_എ", 6: "Malayalam_ഒ",
    # Malayalam Dynamic (7-14)
    7: "Malayalam_അം", 8: "Malayalam_അഃ", 9: "Malayalam_ഈ",
    10: "Malayalam_ഊ", 11: "Malayalam_ഏ", 12: "Malayalam_ഐ",
    13: "Malayalam_ഓ", 14: "Malayalam_ഔ",
    # ISL (15-39)
    15: "ISL_A", 16: "ISL_B", 17: "ISL_C", 18: "ISL_D", 19: "ISL_E",
    20: "ISL_F", 21: "ISL_G", 22: "ISL_H", 23: "ISL_I", 24: "ISL_J",
    25: "ISL_K", 26: "ISL_L", 27: "ISL_M", 28: "ISL_N", 29: "ISL_O",
    30: "ISL_P", 31: "ISL_Q", 32: "ISL_S", 33: "ISL_T", 34: "ISL_U",
    35: "ISL_V", 36: "ISL_W", 37: "ISL_X", 38: "ISL_Y", 39: "ISL_Z"
}


def load_malayalam_static(malayalam_path: str, extractor: MediaPipeExtractor, annotations_path: str = None) -> Dict[int, List[np.ndarray]]:
    """
    Load Malayalam static signs (Classes 0-6).
    Uses annotations.csv to find the correct character folders.
    
    Args:
        malayalam_path: Path to MALAYALAM directory
        extractor: MediaPipeExtractor instance
        annotations_path: Path to annotations.csv (optional)
    
    Returns:
        Dictionary mapping class_id -> list of feature vectors
    """
    logger.info("Loading Malayalam static signs...")
    static_path = Path(malayalam_path) / "Static"
    
    if not static_path.exists():
        logger.error(f"Static directory not found: {static_path}")
        return {}
    
    data_dict = {}
    
    # Get unique static characters from directory or annotations
    static_chars = sorted([d.name for d in static_path.iterdir() if d.is_dir()])
    
    if len(static_chars) == 0:
        logger.warning(f"No character folders found in {static_path}")
        return {}
    
    logger.info(f"Found {len(static_chars)} static character folders: {static_chars}")
    
    # Map character folder names to class IDs
    for class_idx, char_name in enumerate(static_chars):
        if class_idx >= 7:  # Limit to 7 classes for static
            break
        
        char_folder = static_path / char_name
        
        if not char_folder.exists():
            logger.warning(f"Folder not found: {char_folder}")
            continue
        
        # Find all images (both subdirectories and direct files)
        image_files = []
        for item in char_folder.rglob("*"):
            if item.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                image_files.append(item)
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {char_folder}")
            continue
        
        logger.info(f"Processing class {class_idx} ('{char_name}'): {len(image_files)} images")
        
        # Extract features
        features_list = []
        for img_path in tqdm(image_files, desc=f"Class {class_idx}"):
            features = extractor.extract_from_image(str(img_path))
            if features is not None and extractor.validate_features(features):
                features_list.append(features)
        
        data_dict[class_idx] = features_list
        logger.info(f"Class {class_idx}: Extracted {len(features_list)}/{len(image_files)} samples")
    
    return data_dict


def load_malayalam_dynamic(malayalam_path: str, extractor: MediaPipeExtractor, max_seq_len: int = 60) -> Dict[int, List[np.ndarray]]:
    """
    Load Malayalam dynamic signs (Classes 7-14).
    Uses directory structure: Dynamic/[character_name]/[sequence_name]/[frames]
    
    Args:
        malayalam_path: Path to MALAYALAM directory
        extractor: MediaPipeExtractor instance
        max_seq_len: Maximum sequence length
    
    Returns:
        Dictionary mapping class_id -> list of feature sequences
    """
    logger.info("Loading Malayalam dynamic signs...")
    dynamic_path = Path(malayalam_path) / "Dynamic"
    
    if not dynamic_path.exists():
        logger.error(f"Dynamic directory not found: {dynamic_path}")
        return {}
    
    data_dict = {}
    
    # Get unique dynamic characters from directory
    dynamic_chars = sorted([d.name for d in dynamic_path.iterdir() if d.is_dir()])
    
    if len(dynamic_chars) == 0:
        logger.warning(f"No character folders found in {dynamic_path}")
        return {}
    
    logger.info(f"Found {len(dynamic_chars)} dynamic character folders: {dynamic_chars}")
    
    # Map character folder names to class IDs (starting from 7)
    for char_idx, char_name in enumerate(dynamic_chars):
        class_idx = 7 + char_idx
        
        if class_idx >= 15:  # Limit to 8 classes for dynamic (7-14)
            break
        
        char_folder = dynamic_path / char_name
        
        if not char_folder.exists():
            logger.warning(f"Folder not found: {char_folder}")
            continue
        
        # Find all sequence folders
        sequence_folders = [d for d in char_folder.iterdir() if d.is_dir()]
        
        if len(sequence_folders) == 0:
            logger.warning(f"No sequences found in {char_folder}")
            continue
        
        logger.info(f"Processing class {class_idx} ('{char_name}'): {len(sequence_folders)} sequences")
        
        features_list = []
        for seq_folder in tqdm(sequence_folders, desc=f"Class {class_idx}"):
            # Get all frames in sequence
            frame_files = sorted(seq_folder.glob("*.jpg")) + sorted(seq_folder.glob("*.png"))
            
            if len(frame_files) == 0:
                continue
            
            # Extract features from each frame
            seq_features = []
            for frame_path in frame_files:
                features = extractor.extract_from_image(str(frame_path))
                if features is not None and extractor.validate_features(features):
                    seq_features.append(features)
            
            if len(seq_features) > 0:
                # Stack into sequence
                seq_array = np.stack(seq_features, axis=0)
                
                # Truncate or pad to max_seq_len
                if len(seq_array) > max_seq_len:
                    seq_array = seq_array[:max_seq_len]
                elif len(seq_array) < max_seq_len:
                    padding = np.zeros((max_seq_len - len(seq_array), 126), dtype=np.float32)
                    seq_array = np.vstack([seq_array, padding])
                
                features_list.append(seq_array)
        
        data_dict[class_idx] = features_list
        logger.info(f"Class {class_idx}: Extracted {len(features_list)}/{len(sequence_folders)} sequences")
    
    return data_dict


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
    
    # Map alphabet to class ID (A=15, B=16, ..., Z=39, skip R)
    alphabet = "ABCDEFGHIJKLMNOPQSTUVWXYZ"  # No R
    alphabet_to_class = {letter: 15 + idx for idx, letter in enumerate(alphabet)}
    
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
                            features_list.append(features.astype(np.float32))
                except Exception as e:
                    logger.warning(f"Failed to load {npy_file}: {e}")
        
        data_dict[class_idx] = features_list
        logger.info(f"Class {class_idx} ({CLASS_NAMES[class_idx]}): Loaded {len(features_list)} samples")
    
    return data_dict


def apply_augmentation(data_dict: Dict[int, List[np.ndarray]], 
                      augment_count: int = 75,
                      max_seq_len: int = 60) -> Dict[int, List[np.ndarray]]:
    """
    Apply augmentation to all classes.
    
    Args:
        data_dict: Dictionary mapping class_id -> list of samples
        augment_count: Target number of samples for rare classes (7-14)
        max_seq_len: Maximum sequence length
    
    Returns:
        Augmented data dictionary
    """
    logger.info("Applying data augmentation...")
    
    # 1. POV flip for Malayalam static (0-6)
    logger.info("Augmenting Malayalam static (POV flip)...")
    for class_idx in range(7):
        if class_idx in data_dict:
            original_samples = data_dict[class_idx].copy()
            flipped_samples = [horizontal_flip(s) for s in original_samples]
            data_dict[class_idx].extend(flipped_samples)
            logger.info(f"Class {class_idx}: {len(original_samples)} -> {len(data_dict[class_idx])} samples")
    
    # 2. Aggressive augmentation for Malayalam dynamic (7-14)
    logger.info("Augmenting Malayalam dynamic (all techniques)...")
    data_dict = augment_rare_classes(
        data_dict,
        rare_class_ids=list(range(7, 15)),
        target_samples=augment_count
    )
    
    # 3. POV flip for ISL (15-39)
    logger.info("Augmenting ISL (POV flip)...")
    for class_idx in range(15, 40):
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
        'num_classes': 40,
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


def validate_dataset_paths(malayalam_path: str, isl_path: str):
    """
    Validate that dataset paths exist and have correct structure.
    
    Args:
        malayalam_path: Path to MALAYALAM directory
        isl_path: Path to ISL directory
    
    Raises:
        ValueError: If paths are invalid or structure is incorrect
    """
    # Check Malayalam path
    malayalam_base = Path(malayalam_path)
    if not malayalam_base.exists():
        raise ValueError(f"❌ MALAYALAM path does not exist: {malayalam_path}")
    
    static_path = malayalam_base / "Static"
    dynamic_path = malayalam_base / "Dynamic"
    
    if not static_path.exists():
        raise ValueError(f"❌ MALAYALAM/Static directory not found: {static_path}")
    
    if not dynamic_path.exists():
        raise ValueError(f"❌ MALAYALAM/Dynamic directory not found: {dynamic_path}")
    
    # Check ISL path
    isl_base = Path(isl_path)
    if not isl_base.exists():
        raise ValueError(f"❌ ISL path does not exist: {isl_path}")
    
    isl_data_path = isl_base / "data"
    if not isl_data_path.exists():
        raise ValueError(f"❌ ISL/data directory not found: {isl_data_path}")
    
    logger.info(f"✓ MALAYALAM path validated: {malayalam_path}")
    logger.info(f"✓ ISL path validated: {isl_path}")


def validate_loaded_data(all_data: Dict[int, List[np.ndarray]]):
    """
    Validate that all required classes were loaded.
    
    Args:
        all_data: Dictionary mapping class_id -> list of samples
    
    Raises:
        ValueError: If critical data is missing
    """
    malayalam_static_count = sum(len(v) for k, v in all_data.items() if 0 <= k < 7)
    malayalam_dynamic_count = sum(len(v) for k, v in all_data.items() if 7 <= k < 15)
    isl_count = sum(len(v) for k, v in all_data.items() if 15 <= k < 40)
    
    logger.info("")
    logger.info("="*70)
    logger.info("DATA LOADING SUMMARY")
    logger.info("="*70)
    logger.info(f"Malayalam Static (Classes 0-6):   {malayalam_static_count:6d} samples")
    logger.info(f"Malayalam Dynamic (Classes 7-14): {malayalam_dynamic_count:6d} samples")
    logger.info(f"ISL (Classes 15-39):              {isl_count:6d} samples")
    logger.info(f"{'─'*70}")
    logger.info(f"Total:                            {malayalam_static_count + malayalam_dynamic_count + isl_count:6d} samples")
    logger.info("="*70)
    
    # Validation
    if malayalam_static_count == 0:
        logger.error("❌ CRITICAL: No Malayalam Static data loaded!")
        logger.error("   Expected: 1,000+ samples")
        logger.error("   Actual: 0 samples")
        logger.error("   This typically means the MALAYALAM/Static directory is empty")
        raise ValueError("Malayalam Static data not loaded")
    
    if malayalam_dynamic_count == 0:
        logger.error("❌ CRITICAL: No Malayalam Dynamic data loaded!")
        logger.error("   Expected: 1,000+ samples")
        logger.error("   Actual: 0 samples")
        logger.error("   This typically means the MALAYALAM/Dynamic directory is empty")
        raise ValueError("Malayalam Dynamic data not loaded")
    
    if isl_count == 0:
        logger.error("❌ ERROR: No ISL data loaded!")
        logger.error("   Expected: 6,000+ samples")
        logger.error("   Actual: 0 samples")
        raise ValueError("ISL data not loaded")
    
    logger.info("✓ All datasets loaded successfully!")
    logger.info("")
    
    return {
        'malayalam_static': malayalam_static_count,
        'malayalam_dynamic': malayalam_dynamic_count,
        'isl': isl_count,
        'total': malayalam_static_count + malayalam_dynamic_count + isl_count
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess sign language datasets")
    parser.add_argument('--malayalam_path', type=str, required=True,
                       help='Path to MALAYALAM dataset directory')
    parser.add_argument('--isl_path', type=str, required=True,
                       help='Path to ISL dataset directory')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--augment_count', type=int, default=75,
                       help='Target samples per rare class (classes 7-14)')
    parser.add_argument('--max_seq_len', type=int, default=60,
                       help='Maximum sequence length')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='MediaPipe detection confidence threshold')
    
    args = parser.parse_args()
    
    # Validate paths before processing
    try:
        validate_dataset_paths(args.malayalam_path, args.isl_path)
    except ValueError as e:
        logger.error(f"\n{e}\n")
        logger.error("Please ensure:")
        logger.error("  1. MALAYALAM_PATH is set correctly")
        logger.error("  2. ISL_PATH is set correctly")
        logger.error("  3. Both directories contain the required subdirectories")
        logger.error("")
        logger.error("Example:")
        logger.error("  export MALAYALAM_PATH=/path/to/MALAYALAM")
        logger.error("  export ISL_PATH=/path/to/ISL")
        logger.error("  python preprocessing/preprocess.py \\")
        logger.error("    --malayalam_path \"$MALAYALAM_PATH\" \\")
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
    
    # Load datasets
    malayalam_static = load_malayalam_static(args.malayalam_path, extractor)
    malayalam_dynamic = load_malayalam_dynamic(args.malayalam_path, extractor, args.max_seq_len)
    isl_data = load_isl_data(args.isl_path)
    
    # Combine all data
    all_data = {**malayalam_static, **malayalam_dynamic, **isl_data}
    
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
