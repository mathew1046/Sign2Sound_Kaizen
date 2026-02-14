# Dataset Documentation

## Overview

This directory contains the processed datasets for Indian Sign Language (ISL) recognition.

## Dataset Source

### Indian Sign Language (ISL)
- **Classes**: 25 alphabets (A-Z excluding R)
- **Format**: Pre-extracted .npy files (126-dimensional features)
- **Original Location**: `/path/to/datasets/ISL/data/`

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
│   └── ...
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

## Class Mapping

| Class ID | Sign | Language |
|----------|------|----------|
| 0-24 | A-Z (no R) | ISL |

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
   - Convert to 126-dimensional vector

2. **Wrist-Relative Normalization**
   - Normalize coordinates relative to wrist landmark

3. **Data Augmentation**
   - POV horizontal flip (default)

4. **Quality Validation**
   - Remove samples with NaN or Inf values
   - Check for all-zero samples

5. **Sequence Formatting**
   - Pad sequences to max_length=60
   - Handle variable-length sequences

6. **Stratified Splitting**
   - 70% training, 15% validation, 15% test
   - Maintain class distribution across splits
   - Random seed=42 for reproducibility

## Download Instructions

### ISL Dataset
1. Download from official ISL dataset repository
2. Extract to `datasets/ISL/data/`
3. Ensure .npy files are present in subdirectories

## Preprocessing Command

```bash
python preprocessing/preprocess.py \
   --isl_path /path/to/datasets/ISL \
   --output data/processed \
   --max_seq_len 60
```

## Data License

Please refer to original dataset licenses:
- ISL Dataset: [License Information]

Users must comply with original dataset terms and conditions.

## Known Issues

1. **Sequence Length Variation**: Dynamic signs can have variable lengths (5-100 frames). Sequences are normalized to 60 frames.

2. **Missing Hands**: Some images fail hand detection. These samples are excluded during preprocessing.

## Citations

If using these datasets, please cite the original sources:
- [ISL Dataset Citation]
