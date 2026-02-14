# Feature Extraction Modules

This directory contains modules for extracting and processing features from sign language images and videos.

## Modules

### `hand_landmarks.py`
MediaPipe-based hand landmark detection and extraction.

**Key Class: `HandLandmarkDetector`**
- Detects up to 2 hands per image
- Extracts 21 landmarks per hand (42 total)
- Each landmark has (x, y, z) coordinates
- Output: 126-dimensional feature vector

**Landmark Structure:**
```
Hand landmarks (21 per hand):
0: WRIST
1-4: THUMB (CMC, MCP, IP, TIP)
5-8: INDEX_FINGER (MCP, PIP, DIP, TIP)
9-12: MIDDLE_FINGER (MCP, PIP, DIP, TIP)
13-16: RING_FINGER (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)
```

**Usage:**
```python
from hand_landmarks import HandLandmarkDetector

detector = HandLandmarkDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process image
features = detector.process_image(image)  # Shape: (126,)

# Visualize
landmarks_list = detector.detect_hands(image)
annotated = detector.draw_landmarks(image, landmarks_list)

detector.close()
```

### `feature_utils.py`
Utility functions for feature processing and manipulation.

**Key Functions:**
- `pad_sequence()` - Pad sequence to max length
- `truncate_sequence()` - Truncate sequence to max length
- `pad_or_truncate()` - Ensure exact target length
- `create_attention_mask()` - Create padding masks
- `get_sequence_length()` - Count non-padding frames
- `batch_pad_sequences()` - Process multiple sequences
- `split_hands()` - Split into hand1 (63) + hand2 (63)
- `merge_hands()` - Combine hands into 126-dim vector
- `reshape_landmarks()` - Convert to (42, 3) structure
- `flatten_landmarks()` - Convert back to (126,) vector

**Usage:**
```python
from feature_utils import pad_or_truncate, create_attention_mask

# Ensure sequence is exactly 60 frames
padded = pad_or_truncate(features, max_len=60)

# Create attention mask
mask = create_attention_mask(seq_len=actual_length, max_len=60)

# Split hands for analysis
hand1, hand2 = split_hands(features)
```

### `pose_estimation.py` (Placeholder)
Reserved for future body pose integration using MediaPipe Pose.

### `facial_features.py` (Placeholder)
Reserved for future facial expression integration using MediaPipe Face Mesh.

## Feature Vector Structure

### 126-Dimensional Feature Vector

```
[hand1_x0, hand1_y0, hand1_z0,    # Landmark 0: WRIST
 hand1_x1, hand1_y1, hand1_z1,    # Landmark 1: THUMB_CMC
 ...
 hand1_x20, hand1_y20, hand1_z20, # Landmark 20: PINKY_TIP
 hand2_x0, hand2_y0, hand2_z0,    # Hand 2 starts
 ...
 hand2_x20, hand2_y20, hand2_z20] # Hand 2 PINKY_TIP
```

**Dimensions:**
- 2 hands × 21 landmarks × 3 coordinates = 126 features
- Hand 1: indices 0-62
- Hand 2: indices 63-125

**Coordinate System:**
- x: Horizontal position (0 = left, 1 = right)
- y: Vertical position (0 = top, 1 = bottom)
- z: Depth (relative to wrist, usually -0.1 to 0.1)

**Missing Hands:**
- If only 1 hand detected: second hand filled with zeros
- If no hands detected: entire vector is zeros (sample discarded)

## Sequence Handling

### Static Signs
- Single frame per sign
- Stored as (1, 126) then padded to (60, 126)
- Only first frame contains data, rest are zeros

### Dynamic Signs
- Multiple frames per sign (5-100 frames)
- Stored as (num_frames, 126)
- Padded or truncated to (60, 126)
- Actual length tracked for attention masking

## Coordinate Normalization

MediaPipe outputs normalized coordinates in [0, 1] range:
- x, y: Normalized to image dimensions
- z: Normalized to wrist depth

Additional normalization applied:
- Clip to [0, 1] range
- Handle outliers from detection noise

## Processing Pipeline

```
Raw Image/Video
    ↓
MediaPipe Hands Detection
    ↓
Extract 21 landmarks × 2 hands
    ↓
Convert to (x, y, z) coordinates
    ↓
Normalize to [0, 1] range
    ↓
Format as 126-dim vector
    ↓
Pad/truncate to (60, 126)
    ↓
Create attention mask
    ↓
Save as .npy file
```

## MediaPipe Configuration

### For High-Quality Images
```python
HandLandmarkDetector(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### For Challenging Images
```python
HandLandmarkDetector(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lower for better recall
    min_tracking_confidence=0.3
)
```

### For Real-time Video
```python
HandLandmarkDetector(
    static_image_mode=False,  # Enable tracking
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Performance Considerations

**Detection Speed:**
- Single image: ~50-100ms (CPU)
- Single image: ~10-20ms (GPU)
- Batch processing: ~5ms per image (GPU)

**Memory Usage:**
- Single feature vector: 126 × 4 bytes = 504 bytes
- Padded sequence: 60 × 126 × 4 bytes = ~30 KB
- Full dataset: ~27,700 × 30 KB = ~800 MB

## Quality Validation

Features are validated for:
1. No NaN or Inf values
2. Coordinates in valid range [-0.5, 1.5]
3. At least one non-zero landmark
4. Proper shape (126,) or (seq_len, 126)

Invalid samples are automatically filtered during preprocessing.

## Visualization

To visualize landmarks:
```python
from hand_landmarks import HandLandmarkDetector
import cv2

detector = HandLandmarkDetector()
image = cv2.imread('sign.jpg')

landmarks_list = detector.detect_hands(image)
annotated = detector.draw_landmarks(image, landmarks_list)

cv2.imshow('Hand Landmarks', annotated)
cv2.waitKey(0)
```

## Testing

Run module tests:
```bash
# Test hand landmarks
python features/hand_landmarks.py

# Test feature utilities
python features/feature_utils.py
```

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Hand Landmark Model](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model)
- [Hand Landmark Coordinates](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model-bundle)
