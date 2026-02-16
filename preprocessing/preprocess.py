"""
Main Preprocessing Pipeline (STREAMING VERSION - Maximum Memory Efficiency)

Changes:
- Removed unused MediaPipeExtractor to save resources.
- FIXED: Corrected alphabet mapping to skip 'R' properly
- STREAMING: Process and save each class immediately to disk
- Metadata-only splits to prevent OOM
- Final assembly from disk files
"""

import argparse
import gc
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

from augmentation import horizontal_flip
from inference.feature_processing import normalize_landmarks_wrist_relative

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Class mapping (ISL only, A-Z excluding R)
CLASS_NAMES = {
    0: "ISL_A", 1: "ISL_B", 2: "ISL_C", 3: "ISL_D", 4: "ISL_E",
    5: "ISL_F", 6: "ISL_G", 7: "ISL_H", 8: "ISL_I", 9: "ISL_J",
    10: "ISL_K", 11: "ISL_L", 12: "ISL_M", 13: "ISL_N", 14: "ISL_O",
    15: "ISL_P", 16: "ISL_Q", 17: "ISL_S", 18: "ISL_T", 19: "ISL_U",
    20: "ISL_V", 21: "ISL_W", 22: "ISL_X", 23: "ISL_Y", 24: "ISL_Z",
}

def load_dataset_generic(base_path: str, dataset_name: str) -> Dict[int, List[np.ndarray]]:
    """Load a dataset with the standard structure."""
    logger.info(f"Loading {dataset_name} data...")
    data_path = Path(base_path)

    if not data_path.exists():
        logger.error(f"{dataset_name} data directory not found: {data_path}")
        return {}

    # FIXED: Remove 'R' from alphabet before creating the mapping
    alphabet = "ABCDEFGHIJKLMNOPQSTUVWXYZ".replace("R", "")
    alphabet_to_class = {letter: idx for idx, letter in enumerate(alphabet)}
    data_dict = {}

    for letter, class_idx in tqdm(alphabet_to_class.items(), desc=f"Loading {dataset_name}"):
        letter_folder = data_path / letter
        if not letter_folder.exists(): continue

        features_list = []
        for subfolder in sorted(letter_folder.iterdir()):
            if not subfolder.is_dir(): continue
            npy_files = list(subfolder.glob("*.npy"))
            for npy_file in npy_files:
                try:
                    features = np.load(str(npy_file))
                    if features.shape == (126,):
                        if not np.isnan(features).any() and not np.isinf(features).any() and np.any(features):
                            normalized = normalize_landmarks_wrist_relative(features.astype(np.float32))
                            features_list.append(normalized)
                except Exception:
                    pass

        if features_list:
            data_dict[class_idx] = features_list
            logger.info(f"  {letter} (class {class_idx}): Loaded {len(features_list)} samples")

    return data_dict

def load_isl_data(isl_path: str) -> Dict[int, List[np.ndarray]]:
    return load_dataset_generic(str(Path(isl_path) / "data"), "ISL")

def load_data1(data1_path: str) -> Dict[int, List[np.ndarray]]:
    return load_dataset_generic(data1_path, "data1")

def load_data2(data2_path: str) -> Dict[int, List[np.ndarray]]:
    return load_dataset_generic(data2_path, "MP_Data_Normalized")

def process_and_save_class_streaming(
    class_idx: int,
    samples: List[np.ndarray],
    max_seq_len: int,
    temp_dir: Path,
    global_sample_counter: int
) -> Tuple[List[dict], int]:
    """
    Process ONE class completely and save to temp files immediately.
    Returns metadata records and updated sample counter.
    """
    logger.info(f"Processing class {class_idx} ({len(samples)} samples)...")
    
    metadata_records = []
    
    # Process original samples
    for idx, sample in enumerate(samples):
        # Format the sample
        if sample.ndim == 1:
            padded = np.zeros((max_seq_len, 126), dtype=np.float32)
            padded[0] = sample
            formatted = padded
        else:
            formatted = sample
        
        # Ensure wrist-relative normalization before saving
        formatted = normalize_landmarks_wrist_relative(formatted.astype(np.float32))

        # Save immediately to disk
        filename = f"class_{class_idx}_sample_{global_sample_counter}.npy"
        filepath = temp_dir / filename
        np.save(str(filepath), formatted)
        
        # Create metadata
        seq_len = np.count_nonzero(np.any(formatted != 0, axis=1))
        metadata_records.append({
            "sample_id": global_sample_counter,
            "class_idx": class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "file_path": str(filepath),
            "seq_length": seq_len,
            "is_flipped": False
        })
        
        global_sample_counter += 1
        del formatted  # Free memory
    
    logger.info(f"  Saved {len(samples)} original samples")
    
    # Process flipped samples
    for idx, sample in enumerate(samples):
        # Flip
        flipped = horizontal_flip(sample)
        
        # Format
        if flipped.ndim == 1:
            padded = np.zeros((max_seq_len, 126), dtype=np.float32)
            padded[0] = flipped
            formatted = padded
        else:
            formatted = flipped
        
        # Ensure wrist-relative normalization after flip
        formatted = normalize_landmarks_wrist_relative(formatted.astype(np.float32))

        # Save immediately
        filename = f"class_{class_idx}_sample_{global_sample_counter}.npy"
        filepath = temp_dir / filename
        np.save(str(filepath), formatted)
        
        # Create metadata
        seq_len = np.count_nonzero(np.any(formatted != 0, axis=1))
        metadata_records.append({
            "sample_id": global_sample_counter,
            "class_idx": class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "file_path": str(filepath),
            "seq_length": seq_len,
            "is_flipped": True
        })
        
        global_sample_counter += 1
        del flipped, formatted  # Free memory
    
    logger.info(f"  Saved {len(samples)} flipped samples")
    logger.info(f"  ✓ Class {class_idx} complete: {len(metadata_records)} total samples saved")
    
    return metadata_records, global_sample_counter

def process_all_datasets_streaming(
    isl_data: Dict[int, List[np.ndarray]],
    data1: Dict[int, List[np.ndarray]],
    data2: Dict[int, List[np.ndarray]],
    max_seq_len: int,
    temp_dir: Path
) -> pd.DataFrame:
    """
    Process all datasets in streaming fashion - one class at a time.
    Saves data immediately and only keeps metadata in memory.
    """
    logger.info("Starting streaming processing...")
    
    # Combine all class indices
    all_class_indices = set(isl_data.keys()) | set(data1.keys()) | set(data2.keys())
    
    all_metadata = []
    global_sample_counter = 0
    
    for class_idx in sorted(all_class_indices):
        # Combine samples from all datasets for this class
        class_samples = []
        if class_idx in isl_data:
            class_samples.extend(isl_data[class_idx])
        if class_idx in data1:
            class_samples.extend(data1[class_idx])
        if class_idx in data2:
            class_samples.extend(data2[class_idx])
        
        if not class_samples:
            continue
        
        # Process and save this class immediately
        metadata, global_sample_counter = process_and_save_class_streaming(
            class_idx, class_samples, max_seq_len, temp_dir, global_sample_counter
        )
        
        all_metadata.extend(metadata)
        
        # Free memory for this class
        del class_samples, metadata
        gc.collect()
    
    logger.info(f"✓ All classes processed. Total samples: {global_sample_counter}")
    return pd.DataFrame(all_metadata)

def create_splits_from_metadata(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits from metadata."""
    logger.info("Creating train/val/test splits...")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(val_ratio + test_ratio),
        stratify=metadata_df["class_idx"],
        random_state=random_state,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df["class_idx"],
        random_state=random_state,
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

def organize_final_output(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
):
    """Organize the final output structure."""
    logger.info(f"Organizing final output in {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # The files are already saved in features_dir from streaming processing
    # We just need to save the CSV splits
    
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    
    # Save class mapping
    pd.DataFrame(
        [{"class_idx": idx, "class_name": name} for idx, name in CLASS_NAMES.items()]
    ).to_csv(output_dir / "class_mapping.csv", index=False)
    
    # Save summary
    summary = {
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "num_classes": len(CLASS_NAMES),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }
    with open(output_dir / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("✓ Output organized successfully")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isl_path", type=str, required=True)
    parser.add_argument("--data1_path", type=str, required=True)
    parser.add_argument("--data2_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--augment_count", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=60)
    args = parser.parse_args()

    output_path = Path(args.output)
    temp_dir = output_path / "features"  # Save directly to final location
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets (these will be freed class by class)
    isl_data = load_isl_data(args.isl_path)
    data1 = load_data1(args.data1_path)
    data2 = load_data2(args.data2_path)
    
    # Process in streaming fashion (saves to disk immediately, keeps only metadata)
    metadata_df = process_all_datasets_streaming(
        isl_data, data1, data2, args.max_seq_len, temp_dir
    )
    
    # Free the loaded data (we've already saved everything)
    del isl_data, data1, data2
    gc.collect()
    
    # Create splits from metadata only
    train_df, val_df, test_df = create_splits_from_metadata(metadata_df)
    
    # Organize final output
    organize_final_output(train_df, val_df, test_df, output_path)

    logger.info("✓ PREPROCESSING COMPLETE!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)