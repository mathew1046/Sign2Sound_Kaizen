"""
Enhanced Preprocessing with Wrist-Relative Normalization

This script adds wrist-relative normalization to the preprocessing pipeline
to match the real-time inference preprocessing.

Usage:
    python preprocessing/preprocess_enhanced.py \
        --input data/processed \
        --output data/processed_normalized

Author: Team Kaizen
Date: January 2026
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.feature_processing import normalize_landmarks_wrist_relative

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_wrist_normalization_to_dataset(input_dir: Path, output_dir: Path):
    """
    Apply wrist-relative normalization to all features in dataset.
    
    Args:
        input_dir: Input directory containing features/
        output_dir: Output directory for normalized features
    """
    features_dir = input_dir / "features"
    
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return
    
    # Create output directory
    output_features_dir = output_dir / "features"
    output_features_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files
    npy_files = list(features_dir.glob("*.npy"))
    logger.info(f"Found {len(npy_files)} feature files")
    
    # Process each file
    normalized_count = 0
    failed_count = 0
    
    for npy_file in tqdm(npy_files, desc="Normalizing features"):
        try:
            # Load features
            features = np.load(str(npy_file))
            
            # Apply wrist-relative normalization
            normalized_features = normalize_landmarks_wrist_relative(features)
            
            # Save normalized features
            output_file = output_features_dir / npy_file.name
            np.save(str(output_file), normalized_features)
            
            normalized_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to process {npy_file.name}: {e}")
            failed_count += 1
    
    logger.info(f"Normalization complete: {normalized_count} files processed, {failed_count} failed")
    
    # Copy metadata files
    for meta_file in ["class_mapping.csv", "train_split.csv", "val_split.csv", "test_split.csv", 
                     "preprocessing_summary.json"]:
        src = input_dir / meta_file
        if src.exists():
            import shutil
            dst = output_dir / meta_file
            shutil.copy2(str(src), str(dst))
            logger.info(f"Copied {meta_file}")


def verify_normalization(features_dir: Path, num_samples: int = 5):
    """
    Verify normalization by checking sample features.
    
    Args:
        features_dir: Directory containing normalized features
        num_samples: Number of samples to check
    """
    npy_files = list(features_dir.glob("*.npy"))[:num_samples]
    
    logger.info(f"Verifying normalization on {len(npy_files)} samples:")
    
    for npy_file in npy_files:
        features = np.load(str(npy_file))
        
        # Check wrist positions (should be ~0 for normalized features)
        if features.ndim == 1:
            # Single frame
            wrist1 = features[0:3]  # Hand 1 wrist
            wrist2 = features[63:66]  # Hand 2 wrist
            
            logger.info(f"{npy_file.name}:")
            logger.info(f"  Hand 1 wrist: {wrist1}")
            logger.info(f"  Hand 2 wrist: {wrist2}")
        else:
            # Sequence
            wrist1_first = features[0, 0:3]
            wrist2_first = features[0, 63:66]
            
            logger.info(f"{npy_file.name} (first frame):")
            logger.info(f"  Hand 1 wrist: {wrist1_first}")
            logger.info(f"  Hand 2 wrist: {wrist2_first}")


def main():
    parser = argparse.ArgumentParser(description="Apply wrist-relative normalization to dataset")
    parser.add_argument('--input', type=str, default='data/processed',
                       help='Input directory containing features')
    parser.add_argument('--output', type=str, default='data/processed_normalized',
                       help='Output directory for normalized features')
    parser.add_argument('--verify', action='store_true',
                       help='Verify normalization after processing')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Apply normalization
    apply_wrist_normalization_to_dataset(input_dir, output_dir)
    
    # Verify if requested
    if args.verify:
        logger.info("\nVerifying normalization:")
        verify_normalization(output_dir / "features", num_samples=5)
    
    logger.info("\nComplete! Update config.yaml to use normalized data:")
    logger.info(f"  train_csv: {output_dir}/train_split.csv")
    logger.info(f"  val_csv: {output_dir}/val_split.csv")
    logger.info(f"  test_csv: {output_dir}/test_split.csv")


if __name__ == "__main__":
    main()
