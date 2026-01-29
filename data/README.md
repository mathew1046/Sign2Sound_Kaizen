# Dataset Documentation

## Overview

This directory contains the processed datasets for sign language recognition.

## Dataset Sources

### Malayalam Sign Language
- **Static Signs**: 7 classes (അ, ആ, ഇ, ഈ, ഉ, ഏ, ഐ)
- **Dynamic Signs**: 8 classes (ഒ, ഓ, ഔ, ക, ഖ, ഗ, ഘ, ങ)
- **Format**: JPG images (static), sequence of frames (dynamic)
- **Original Location**: `/path/to/datasets/MALAYALAM/`

### Indian Sign Language (ISL)
- **Classes**: 25 alphabets (A-Z excluding R)
- **Format**: Pre-extracted .npy files (126-dimensional features)
- **Original Location**: `/path/to/datasets/ISL/data/`

## Expected Raw Dataset Structure

```
datasets/
├── MALAYALAM/
│   ├── Static/
│   │   ├── Character_1/        # Class 0: അ
│   │   │   ├── image_001.jpg
│   │   │   ├── image_002.jpg
│   │   │   └── ... (~150 images)
│   │   ├── Character_2/        # Class 1: ആ
│   │   └── ... (7 total)
│   ├── Dynamic/
│   │   ├── Character_1/        # Class 7: ഒ
│   │   │   ├── sequence_001/
│   │   │   │   ├── frame_001.jpg
│   │   │   │   ├── frame_002.jpg
│   │   │   │   └── ... (variable frames)
│   │   │   ├── sequence_002/
│   │   │   └── ...
│   │   ├── Character_2/        # Class 8: ഓ
│   │   └── ... (8 total)
│   ├── annotations.csv
│   └── README.txt
│
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
│   └── ... (96,092+ files)
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

## Class Mapping

| Class ID | Category | Sign | Language |
|----------|----------|------|----------|
| 0 | Static | അ | Malayalam |
| 1 | Static | ആ | Malayalam |
| 2 | Static | ഇ | Malayalam |
| 3 | Static | ഈ | Malayalam |
| 4 | Static | ഉ | Malayalam |
| 5 | Static | ഏ | Malayalam |
| 6 | Static | ഐ | Malayalam |
| 7 | Dynamic | ഒ | Malayalam |
| 8 | Dynamic | ഓ | Malayalam |
| 9 | Dynamic | ഔ | Malayalam |
| 10 | Dynamic | ക | Malayalam |
| 11 | Dynamic | ഖ | Malayalam |
| 12 | Dynamic | ഗ | Malayalam |
| 13 | Dynamic | ഘ | Malayalam |
| 14 | Dynamic | ങ | Malayalam |
| 15-39 | Static | A-Z (no R) | ISL |

## Dataset Statistics

### Before Preprocessing
- Malayalam Static (0-6): ~150 samples per class = 1,050 total
- Malayalam Dynamic (7-14): ~5-10 valid samples per class = ~60 total (96% fail MediaPipe)
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
   - Classes 0-6: POV flip only
   - Classes 7-14: All 6 augmentation techniques (50-100 samples per class)
   - Classes 15-39: POV flip only

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

### Malayalam Dataset
1. Contact dataset provider or access from authorized source
2. Extract to `datasets/MALAYALAM/`
3. Verify structure matches expected format

### ISL Dataset
1. Download from official ISL dataset repository
2. Extract to `datasets/ISL/data/`
3. Ensure .npy files are present in subdirectories

## Preprocessing Command

```bash
python preprocessing/preprocess.py \
    --malayalam_path /path/to/datasets/MALAYALAM \
    --isl_path /path/to/datasets/ISL \
    --output data/processed \
    --augment_count 75 \
    --max_seq_len 60
```

## Data License

Please refer to original dataset licenses:
- Malayalam Sign Language: [License Information]
- ISL Dataset: [License Information]

Users must comply with original dataset terms and conditions.

## Known Issues

1. **Classes 7-14 Quality**: Original Malayalam dynamic signs have very low MediaPipe detection rate (~4% success). Aggressive augmentation compensates for this.

2. **Sequence Length Variation**: Dynamic signs have variable lengths (5-100 frames). Sequences are normalized to 60 frames.

3. **Missing Hands**: Some images fail hand detection. These samples are excluded during preprocessing.

4. **Class Imbalance**: ISL has ~10x more samples than Malayalam. Class weights are used during training to handle this.

## Citations

If using these datasets, please cite the original sources:
- [Malayalam Dataset Citation]
- [ISL Dataset Citation]
