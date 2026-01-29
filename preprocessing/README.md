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
```bash
python preprocessing/preprocess.py \
    --malayalam_path /path/to/MALAYALAM \
    --isl_path /path/to/ISL \
    --output data/processed \
    --augment_count 75 \
    --max_seq_len 60 \
    --min_confidence 0.3
```

**Arguments:**
- `--malayalam_path`: Path to Malayalam dataset directory
- `--isl_path`: Path to ISL dataset directory
- `--output`: Output directory for processed data (default: `data/processed`)
- `--augment_count`: Target number of samples for rare classes 7-14 (default: 75)
- `--max_seq_len`: Maximum sequence length (default: 60)
- `--min_confidence`: MediaPipe detection confidence (default: 0.3)

### `extract_features.py`
MediaPipe-based feature extraction module.

**Key Classes:**
- `MediaPipeExtractor`: Extract 126-dimensional features from images/videos
  - `extract_from_image()`: Single image extraction
  - `extract_from_sequence()`: Video sequence extraction
  - `validate_features()`: Quality validation

**Feature Structure:**
```
126 dimensions = 2 hands × 21 landmarks × 3 coordinates (x, y, z)
- Hand 1: landmarks 0-20 (63 features)
- Hand 2: landmarks 0-20 (63 features)
```

### `augmentation.py`
Data augmentation techniques for improving model robustness.

**Techniques:**
1. **Gaussian Noise** (`add_gaussian_noise`): σ ∈ {0.01, 0.02, 0.03}
2. **Scale Variation** (`scale_variation`): scale ∈ [0.85, 1.15]
3. **Translation** (`translate`): ±0.1 in x and y
4. **Rotation** (`rotate`): ±15 degrees
5. **Horizontal Flip** (`horizontal_flip`): POV augmentation
6. **Temporal Speed** (`temporal_speed`): 0.8-1.2x speed (dynamic signs only)

**Usage:**
```python
from augmentation import augment_sample, horizontal_flip

# Generate 10 augmented versions
augmented = augment_sample(original_data, num_augmentations=10)

# Apply POV flip
flipped = horizontal_flip(original_data)
```

## MediaPipe Configuration

### Settings for Malayalam Static (Classes 0-6)
```python
MediaPipeExtractor(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Settings for Malayalam Dynamic (Classes 7-14)
```python
MediaPipeExtractor(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lower for better recall
    min_tracking_confidence=0.3
)
```

## Augmentation Strategy

### Malayalam Static (Classes 0-6)
- **Original**: ~150 samples per class
- **Technique**: Horizontal flip only (POV)
- **Result**: ~300 samples per class

### Malayalam Dynamic (Classes 7-14) - PROBLEMATIC
- **Original**: ~5-10 valid samples per class (96% fail MediaPipe)
- **Techniques**: ALL 6 augmentation methods
- **Result**: 75 samples per class (configurable)
- **Critical**: Without augmentation, these classes have 0% accuracy

### ISL (Classes 15-39)
- **Original**: ~500 samples per class
- **Technique**: Horizontal flip only (POV)
- **Result**: ~1,000 samples per class

## Output Structure

After preprocessing, the following structure is created:

```
data/processed/
├── features/
│   ├── class_0_sample_0.npy       # (60, 126)
│   ├── class_0_sample_1.npy
│   └── ... (~27,700+ files)
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

### CSV Format

**Split CSV columns:**
- `sample_id`: Unique sample identifier
- `class_idx`: Class index (0-39)
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

### Issue 1: Low MediaPipe Detection Rate for Classes 7-14
**Problem**: ~96% of Malayalam dynamic images fail hand detection.

**Solution**: 
- Use `min_detection_confidence=0.3` (lower threshold)
- Apply aggressive augmentation to valid samples
- Generate 50-100 samples per class from ~5 originals

### Issue 2: Variable Sequence Lengths
**Problem**: Dynamic signs have 5-100 frames per sequence.

**Solution**:
- Pad short sequences to `max_seq_len=60` with zeros
- Truncate long sequences to `max_seq_len=60`
- Store original sequence length for attention masks

### Issue 3: Class Imbalance
**Problem**: ISL has 10x more samples than Malayalam.

**Solution**:
- Use weighted cross-entropy loss during training
- Class weights calculated as: `total / (num_classes * class_count)`
- Malayalam classes get ~125x higher weight than ISL

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
