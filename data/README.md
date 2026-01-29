# Dataset Documentation

## Overview

This directory contains the processed datasets for Indian Sign Language (ISL) recognition.

## Dataset Source

### Indian Sign Language (ISL)
- **Classes**: 25 alphabets (A-Z excluding R)
- **Format**: Pre-extracted .npy files (126-dimensional features)
- **Original Location**: Set via `ISL_PATH` environment variable

## Expected Raw Dataset Structure

```
datasets/
└── ISL/
    └── data/
        ├── A/
        │   ├── 0/
        │   │   ├── 0.npy
        │   │   ├── 1.npy
        │   │   └── ...
        │   ├── 1/
        │   └── ... (120 subfolders, 0-119)
        ├── B/
        └── ... (25 classes total)
```

## Processed Dataset Structure

After running preprocessing, the following structure will be created:

```
data/processed/
├── features/
│   ├── class_0_sample_0.npy    # Shape: (seq_len, 126)
│   ├── class_0_sample_1.npy
│   └── ... (feature files for 25 ISL classes)
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

## Class Mapping

| Class ID | Sign | Language |
|----------|------|----------|
| 0-24 | A-Z (excluding R) | ISL |

Specific mapping:
- Class 0: A
- Class 1: B
- ...
- Class 16: Q
- Class 17: S (R skipped)
- ...
- Class 24: Z

## Dataset Statistics
- ISL (15-39): ~500 samples per class = 12,500 total
- **Total Raw**: ~13,610 samples

### After Preprocessing (with Augmentation)
- Malayalam Static (0-6): ~300 samples per class (with POV flip) = 2,100 total
- Malayalam Dynamic (7-14): ~75 samples per class (heavy augmentation) = 600 total
- ISL (15-39): ~1,000 samples per class (with POV flip) = 25,000 total
- **Total Processed**: ~27,700+ samples (adjustable based on augmentation settings)

### Train/Val/Test Split (70/15/15)
- Training: 19,390 samples (70%)
- Validation: 4,155 samples (15%)
- Testing: 4,155 samples (15%)

## Feature Format

Each processed sample is saved as a `.npy` file with shape:
- **Static signs**: (60, 126) - padded with zeros
- **Dynamic signs**: (60, 126) - truncated or padded

Feature vector structure (126 dimensions):
```
[hand1_x0, hand1_y0, hand1_z0,    # Wrist (landmark 0)
 hand1_x1, hand1_y1, hand1_z1,    # Thumb CMC (landmark 1)
 ...
 hand1_x20, hand1_y20, hand1_z20, # Pinky tip (landmark 20)
 hand2_x0, hand2_y0, hand2_z0,    # Second hand starts
 ...
 hand2_x20, hand2_y20, hand2_z20] # Second hand pinky tip
```

If only one hand is detected, the second hand features are filled with zeros.

## Preprocessing Steps

1. **Feature Extraction**
   - MediaPipe Hands with `min_detection_confidence=0.3`
   - Extract 21 landmarks per hand (42 total)
   - Normalize to [0, 1] range
   - Convert to 126-dimensional vector

2. **Data Augmentation**
   - Horizontal flip (POV flip) to double dataset size
   - Preserves hand laterality

3. **Quality Validation**
   - Remove samples with NaN or Inf values
   - Check for all-zero samples
   - Validate coordinate ranges

4. **Sequence Formatting**
   - Pad sequences to max_length=60
   - Create attention masks for valid frames
   - Handle variable-length sequences

5. **Stratified Splitting**
   - 70% training, 15% validation, 15% test
   - Maintain class distribution across splits
   - Random seed=42 for reproducibility

## Download Instructions

### ISL Dataset
1. Download from official ISL dataset repository
2. Extract to location of your choice
3. Set environment variable: `export ISL_PATH=/path/to/datasets/ISL/data`
4. Ensure .npy files are present in subdirectories (A/, B/, C/, etc.)

## Preprocessing Command

```bash
export ISL_PATH=/path/to/datasets/ISL/data
python preprocessing/preprocess.py \
    --isl_path "$ISL_PATH" \
    --output data/processed \
    --max_seq_len 60
```

Or use the automated script:
```bash
./run_all.sh
```

## Data License

Please refer to the ISL dataset license and comply with original terms and conditions.

## Known Issues

1. **Sequence Length Variation**: Different signs have variable sequence lengths. All sequences are normalized to 60 frames via padding.

2. **Missing Hands**: Some samples may fail hand detection. These samples are excluded during preprocessing.

## Citations

If using this dataset, please cite the original ISL dataset source.
