# Preprocessing Pipeline

This module handles the complete preprocessing pipeline for sign language recognition.

## Overview

The preprocessing pipeline consists of several stages:

1. **Feature Extraction**: Extract hand landmarks using MediaPipe
2. **Data Augmentation**: Apply augmentation techniques to improve robustness
3. **Quality Validation**: Filter invalid samples
4. **Data Splitting**: Create stratified train/val/test splits
5. **Data Saving**: Save processed features and metadata

## Modules

### `preprocess.py`
Main preprocessing script that orchestrates the entire pipeline.

**Usage:**
# Preprocessing Module

This module handles preprocessing of multiple sign language datasets (ISL, data1, MP_Data_Normalized).

## Usage

```bash
python preprocessing/preprocess.py \
    --isl_path /path/to/ISL \
    --data1_path ./data1/MP_Data_Normalized \
    --data2_path ./MP_Data_Normalized \
    --output data/processed
```

## Arguments

- `--isl_path`: Path to ISL dataset directory (required)
- `--data1_path`: Path to data1/MP_Data_Normalized directory (required)
- `--data2_path`: Path to MP_Data_Normalized directory (required)
- `--output`: Output directory for processed data (default: `data/processed`)
- `--augment_count`: Unused for ISL-only pipeline (reserved)
- `--max_seq_len`: Maximum sequence length (default: 60)
- `--min_confidence`: MediaPipe detection confidence threshold (default: 0.3)

## Preprocessing Steps

1. Load all three datasets (ISL, data1, MP_Data_Normalized)
2. Exclude letter R (not available in ISL dataset)
3. Extract MediaPipe hand landmarks
4. Apply wrist-relative normalization
5. Combine all datasets
6. Apply POV flip augmentation
7. Create train/val/test splits
8. Save processed features

## Data Augmentation

- POV horizontal flip (default)

## Class Distribution

- Classes 0-24 (A-Z excluding R, 25 total letters)
- Data from 3 different datasets combined

## MediaPipe Settings

```python
MediaPipeExtractor(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Augmentation Strategy

- POV horizontal flip (default)

## Output Structure

After preprocessing, the following structure is created:

```
data/processed/
├── features/
│   ├── class_0_sample_0.npy       # (60, 126)
│   ├── class_0_sample_1.npy
│   └── ... (combined samples from all 3 datasets)
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

### CSV Format

**Split CSV columns:**
- `sample_id`: Unique sample identifier
- `class_idx`: Class index (0-24, excluding R)
- `class_name`: Human-readable class name
- `file_path`: Path to .npy feature file
- `seq_length`: Number of valid frames in sequence

### Summary JSON

Contains:
- Total samples per split
- Class distribution
- Feature dimensions
- Processing statistics

## Quality Validation

Samples are validated to ensure:
1. No NaN or Inf values
2. No all-zero features
3. Coordinates in valid range [-0.5, 1.5]
4. At least one hand detected

Invalid samples are automatically filtered out.

## Processing Time

Approximate processing times (varies by hardware):

| Stage | CPU | GPU |
|-------|-----|-----|
| Feature Extraction | 30-45 min | 15-20 min |
| Augmentation | 15-20 min | 10-15 min |
| Splitting & Saving | 5-10 min | 5-10 min |
| **Total** | **50-75 min** | **30-45 min** |

## Known Issues & Solutions

### Issue 1: Variable Sequence Lengths
**Problem**: Dynamic signs can have 5-100 frames per sequence.

**Solution**:
- Pad short sequences to `max_seq_len=60` with zeros
- Truncate long sequences to `max_seq_len=60`
- Store original sequence length for attention masks

### Issue 2: Class Imbalance
**Problem**: Some ISL letters may have fewer samples.

**Solution**:
- Use weighted cross-entropy loss during training
- Class weights calculated as: `total / (num_classes * class_count)`

### Issue 3: Letter R Exclusion
**Problem**: Letter R is not available in the ISL dataset.

**Solution**:
- Automatically skip letter R during loading
- Total classes reduced to 25 (A-Z excluding R)
- Class indices: A=0, B=1, ..., Q=16, S=17, ..., Z=24

## Troubleshooting

**MediaPipe fails to import:**
```bash
pip install mediapipe --upgrade
```

**Out of memory:**
- Process datasets sequentially instead of loading all at once
- Reduce batch size for feature extraction
- Use generator functions instead of loading to memory

**Augmentation produces invalid samples:**
- Check `validate_augmented_sample()` function
- Ensure clipping to [0, 1] range is applied
- Verify rotation center calculation for sparse landmarks

## Performance Optimization

Tips for faster preprocessing:
1. Use SSD for dataset storage (45x faster than Google Drive)
2. Increase MediaPipe confidence threshold for clean datasets
3. Reduce augmentation count for well-represented classes
4. Use multiprocessing for batch feature extraction (optional)
5. Cache extracted features before augmentation

## Testing

Run module tests:
```bash
# Test augmentation
python preprocessing/augmentation.py

# Test feature extraction
python preprocessing/extract_features.py
```

## References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [Hand Landmark Model](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model)
