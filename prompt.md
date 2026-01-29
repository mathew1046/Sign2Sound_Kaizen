**PROMPT FOR OPENCODE:**

Create a complete, production-ready sign language recognition system with team name "Kaizen" following the exact folder structure provided. Generate ALL files with working, executable code that can be run immediately to preprocess data, train models, and generate results. Every file should be fully functional with proper imports, error handling, and documentation.

---

## PROJECT OVERVIEW

**Team Name:** Kaizen  
**Project:** BiLSTM-based Sign Language Recognition (Malayalam + ISL)  
**Goal:** Achieve 98%+ test accuracy with robust preprocessing and data augmentation to fix problematic classes 7-14  
**Framework:** PyTorch  
**Key Innovation:** Aggressive data augmentation for under-represented classes

---

## DATASET STRUCTURE (EXACT AS PROVIDED)

```
/path/to/datasets/
├── MALAYALAM/
│   ├── Static/
│   │   ├── Character_1/
│   │   │   ├── image_001.jpg
│   │   │   ├── image_002.jpg
│   │   │   └── ...
│   │   ├── Character_2/
│   │   └── ... (7 characters: 0-6)
│   ├── Dynamic/
│   │   ├── Character_1/
│   │   │   ├── sequence_001/
│   │   │   │   ├── frame_001.jpg
│   │   │   │   ├── frame_002.jpg
│   │   │   │   └── ...
│   │   │   ├── sequence_002/
│   │   │   └── ...
│   │   ├── Character_2/
│   │   └── ... (8 characters: 7-14)
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
        │   └── ... (0-119 folders)
        ├── B/
        └── ... (25 classes: A-Z, excluding R)
```

**Class Mapping:**
- Malayalam Static: Classes 0-6 (അ, ആ, ഇ, ഈ, ഉ, ഏ, ഐ)
- Malayalam Dynamic: Classes 7-14 (ഒ, ഓ, ഔ, ക, ഖ, ഗ, ഘ, ങ) - **PROBLEMATIC - need heavy augmentation**
- ISL: Classes 15-39 (25 alphabets: A-Z excluding R)
- **Total: 40 classes**

**Known Issues:**
- Classes 7-14 have severe quality issues (96% samples fail MediaPipe extraction)
- Target: Generate 50-100 augmented samples per class for 7-14
- ISL classes: ~500 samples each (keep as is)
- Malayalam static (0-6): ~150 samples each (keep as is)

---

## MODEL ARCHITECTURE SPECIFICATIONS

**BiLSTM Architecture (2.53M parameters, 12.8 MB):**

```python
# Exact architecture that achieved 98.41% test accuracy
BiLSTMClassifier(
    input_size=126,        # 2 hands × 21 landmarks × 3 coords
    hidden_size=256,       # Hidden units per direction
    num_layers=2,          # LSTM layers
    num_classes=40,        # Output classes
    dropout=0.3,           # Dropout rate
    bidirectional=True     # Forward + backward
)

# Layer Structure:
# Input: (batch, 60, 126)
# → LSTM: (batch, 60, 512) [256 × 2 directions]
# → Last Hidden: (batch, 512)
# → Dropout(0.3)
# → FC1: (batch, 512) → (batch, 256) + ReLU + Dropout(0.3)
# → FC2: (batch, 256) → (batch, 128) + ReLU + Dropout(0.3)
# → FC3: (batch, 128) → (batch, 40)
# Output: (batch, 40) logits
```

**Key Features:**
- Packed sequences for variable-length handling
- Gradient clipping (max_norm=1.0)
- Attention masks for padding
- Real-time capable: 8ms inference, 35 FPS

---

## PREPROCESSING PIPELINE (CRITICAL)

### Step 1: MediaPipe Feature Extraction

**For Malayalam Images (jpg):**
```python
# Use relaxed settings for problematic classes 7-14
mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lower for better recall
    min_tracking_confidence=0.3
)
```

**For ISL (.npy files):**
- Already preprocessed, keep as is
- Shape: (126,) for each sample
- Direct load with np.load()

**Feature Structure:**
```python
# 126 features per frame/image:
[hand1_x0, hand1_y0, hand1_z0,  # Landmark 0 (wrist)
 hand1_x1, hand1_y1, hand1_z1,  # Landmark 1
 ...
 hand1_x20, hand1_y20, hand1_z20,  # Landmark 20 (pinky tip)
 hand2_x0, hand2_y0, hand2_z0,     # Hand 2 starts
 ...
 hand2_x20, hand2_y20, hand2_z20]

# If only 1 hand: fill hand2 with zeros
# If no hands: mark as invalid, skip
```

### Step 2: Data Augmentation (AGGRESSIVE for Classes 7-14)

**Augmentation Techniques:**

1. **Gaussian Noise** (σ=0.01, 0.02, 0.03):
```python
noise = np.random.normal(0, sigma, data.shape)
augmented = data + noise
augmented = np.clip(augmented, 0, 1)
```

2. **Scale Variation** (0.85-1.15):
```python
scale = np.random.uniform(0.85, 1.15)
augmented = data * scale
augmented = np.clip(augmented, 0, 1)
```

3. **Translation** (±0.1):
```python
tx = np.random.uniform(-0.1, 0.1)
ty = np.random.uniform(-0.1, 0.1)
augmented[:, 0::3] += tx  # x coords
augmented[:, 1::3] += ty  # y coords
augmented = np.clip(augmented, 0, 1)
```

4. **Rotation** (±15 degrees):
```python
angle = np.random.uniform(-15, 15)
# Apply 2D rotation to (x, y) coordinates
# Keep z unchanged
```

5. **Horizontal Flip** (POV augmentation):
```python
augmented[:, 0::3] = 1.0 - data[:, 0::3]  # Mirror x-axis
```

6. **Temporal Speed Variation** (0.8-1.2x for dynamic):
```python
# Interpolate to simulate speed change
# Only for dynamic signs (classes 7-14)
```

**Augmentation Strategy:**
- Classes 0-6 (Malayalam static): POV flip only (already ~150 samples)
- **Classes 7-14 (Malayalam dynamic): ALL augmentations, generate 50-100 per class**
- Classes 15-39 (ISL): POV flip only (already ~500 samples)

### Step 3: Data Cleaning

**Quality Checks:**
```python
def is_valid(data):
    # Check 1: No NaN
    if np.isnan(data).any():
        return False
    # Check 2: No Inf
    if np.isinf(data).any():
        return False
    # Check 3: Not all zeros
    if not np.any(data):
        return False
    # Check 4: In valid range
    if np.any(data < -0.5) or np.any(data > 1.5):
        return False
    return True
```

### Step 4: Sequence Formatting

**Static Signs:**
- Input: (1, 126) single frame
- Pad to: (60, 126) during training
- Mask: [False, True, True, ..., True] (only first frame valid)

**Dynamic Signs:**
- Input: (T, 126) where T = number of frames
- If T < 60: Pad with zeros to (60, 126)
- If T > 60: Truncate to (60, 126)
- Mask: Track valid frame count

### Step 5: Train/Val/Test Split

**Stratified Split (70/15/15):**
```python
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class_idx'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class_idx'], random_state=42)
```

**Save to:**
```
data/processed/
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_mapping.csv
└── preprocessing_summary.json
```

---

## TRAINING CONFIGURATION

**Hyperparameters (config.yaml):**
```yaml
# Model
model:
  name: BiLSTM
  input_size: 126
  hidden_size: 256
  num_layers: 2
  num_classes: 40
  dropout: 0.3
  max_seq_len: 60

# Training
training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: AdamW
  scheduler: ReduceLROnPlateau
  lr_patience: 5
  lr_factor: 0.5
  min_lr: 0.000001
  gradient_clip: 1.0

# Early Stopping
early_stopping:
  patience: 7
  min_delta: 0.001

# Data Loading
data:
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

# Checkpointing
checkpointing:
  save_interval: 5  # Save every 5 epochs
  save_best: true
  save_final: true

# Device
device: cuda  # or cpu
seed: 42
```

**Loss Function:**
```python
# Weighted CrossEntropyLoss for class imbalance
# Weight calculation: total_samples / (num_classes * class_count)
# Malayalam classes get ~155x higher weight than ISL
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Class Weights:**
- ISL classes (15-39): ~0.02 each
- Malayalam static (0-6): ~0.5 each
- Malayalam dynamic (7-14): ~2.5 each (after augmentation)
- Ratio: ~125x between Malayalam and ISL

**Expected Performance:**
- Training time: 5-15 minutes (depends on GPU)
- Convergence: ~20-25 epochs with early stopping
- Test accuracy: 98%+ overall
- Classes 7-14: 80-90% (up from 0% with augmentation)

---

## INFERENCE PIPELINE

**Real-time Demo Requirements:**
- MediaPipe Hands for live video
- Prediction smoothing (5-frame history)
- Confidence threshold: 0.7
- FPS target: 25-30
- Latency: <50ms end-to-end

**Single Image Inference:**
```python
def predict(image_path, model):
    # 1. Load image
    # 2. Extract MediaPipe landmarks → 126 features
    # 3. Pad to (60, 126)
    # 4. Model inference
    # 5. Return class + confidence
    return predicted_class, confidence
```

---

## FILE-BY-FILE REQUIREMENTS

### ROOT LEVEL

**README.md:**
- Project overview with architecture diagram
- Installation instructions
- Quick start guide
- Dataset setup instructions
- Training command examples
- Inference examples
- Results summary (accuracy, F1, etc.)
- Citation and license

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
mediapipe>=0.10.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
scikit-learn>=1.0.0
pyyaml>=6.0
pyttsx3>=2.90  # For TTS
pillow>=9.0.0
```

**LICENSE:**
- MIT License

**.gitignore:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Data
data/raw/*
data/processed/*
!data/processed/.gitkeep
checkpoints/*.pth
checkpoints/*.pt
!checkpoints/README.md

# Results
results/*.png
results/*.json
results/*.csv
results/sample_outputs/*

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### data/

**data/README.md:**
- Dataset sources and licenses
- Download instructions for Malayalam and ISL datasets
- Expected folder structure
- Class mapping explanation
- Statistics (samples per class, split ratios)

**data/statistics.txt:**
```
Total Samples: 96,092
Total Classes: 40
Train: 67,805 (70%)
Val: 13,756 (15%)
Test: 14,531 (15%)

Malayalam (Classes 0-14):
  Static (0-6): ~150 samples each
  Dynamic (7-14): ~50-100 samples each (augmented)
  
ISL (Classes 15-39):
  All classes: ~500 samples each
```

### preprocessing/

**preprocessing/preprocess.py:**
- Main script to run entire preprocessing pipeline
- Command: `python preprocessing/preprocess.py --malayalam_path /path --isl_path /path --output data/processed`
- Functions:
  - `load_malayalam_images()` - Load jpg images from Malayalam/Static and Dynamic
  - `load_isl_npy()` - Load .npy files from ISL/data/
  - `extract_mediapipe_features()` - Apply MediaPipe to images
  - `augment_rare_classes()` - Generate augmented samples for classes 7-14
  - `create_splits()` - Stratified train/val/test split
  - `save_processed_data()` - Save to .npy files and CSVs
  - `generate_statistics()` - Create summary statistics

**preprocessing/augmentation.py:**
- `add_gaussian_noise(data, sigma)` - Add noise augmentation
- `scale_variation(data, scale_range)` - Scale augmentation
- `translate(data, tx, ty)` - Translation augmentation
- `rotate(data, angle)` - Rotation augmentation
- `horizontal_flip(data)` - POV augmentation
- `temporal_speed(data, speed_factor)` - Speed variation for dynamic
- `augment_sample(data, num_aug, techniques)` - Generate N augmented versions
- Complete implementation with proper clipping and validation

**preprocessing/extract_features.py:**
- `MediaPipeExtractor` class:
  - `__init__()` - Initialize MediaPipe Hands
  - `extract_from_image(image_path)` - Extract 126 features from single image
  - `extract_from_sequence(frame_paths)` - Extract from video frames (dynamic)
  - `validate_features(features)` - Check for NaN/Inf/zeros
- Helper functions for batch processing

**preprocessing/README.md:**
- Preprocessing pipeline explanation
- MediaPipe configuration details
- Augmentation techniques documentation
- Usage examples
- Expected output format

### features/

**features/hand_landmarks.py:**
- `HandLandmarkDetector` class using MediaPipe
- `detect_hands(image)` - Returns 42 landmarks (2 hands × 21 points)
- `normalize_landmarks(landmarks)` - Normalize to [0,1] range
- `format_features(landmarks)` - Convert to 126-dim vector
- Handles missing hands (fill with zeros)

**features/pose_estimation.py:**
- Placeholder (not used in this project, but required by structure)
- Can be used for future body pose integration

**features/facial_features.py:**
- Placeholder (not used in this project)
- For future facial expression integration

**features/feature_utils.py:**
- `pad_sequence(data, max_len)` - Pad to max length
- `truncate_sequence(data, max_len)` - Truncate if too long
- `create_attention_mask(seq_len, max_len)` - Create mask for padding
- `batch_process(image_paths, extractor)` - Batch feature extraction

**features/README.md:**
- MediaPipe Hands documentation
- Landmark indices and meanings
- Feature vector structure explanation
- Coordinate system details

### models/

**models/model.py:**
```python
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sign language recognition
    Architecture: 2-layer BiLSTM + 3 FC layers
    Parameters: 2.53M
    """
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, 
                 num_classes=40, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, data, seq_lengths):
        batch_size = data.size(0)
        
        # Sort by length for packing
        seq_lengths_cpu = seq_lengths.cpu()
        seq_lengths_sorted, sorted_idx = seq_lengths_cpu.sort(descending=True)
        data_sorted = data[sorted_idx]
        
        # Pack sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            data_sorted, seq_lengths_sorted, batch_first=True, enforce_sorted=True
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Get last hidden state (both directions)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Unsort
        _, unsort_idx = sorted_idx.sort()
        last_hidden = last_hidden[unsort_idx]
        
        # FC layers
        x = self.dropout(last_hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits
```

**models/custom_layers.py:**
- Placeholder for any custom layers (not needed for basic BiLSTM)

**models/loss.py:**
- `WeightedCrossEntropyLoss` - Wrapper for class-weighted loss
- `calculate_class_weights(train_df)` - Compute inverse frequency weights

**models/README.md:**
- Architecture diagram and explanation
- Layer dimensions and parameter counts
- Forward pass description
- Training considerations

### training/

**training/train.py:**
- Main training script with complete pipeline
- `train_one_epoch()` - Single epoch training loop
- `validate()` - Validation loop
- `train_model()` - Master training function
- Command: `python training/train.py --config training/config.yaml`
- Implements:
  - Progress bars (tqdm)
  - Loss/accuracy tracking
  - Learning rate scheduling
  - Early stopping
  - Checkpointing
  - Logging
- Saves:
  - `checkpoints/best_model.pth`
  - `checkpoints/final_model.pth`
  - `results/metrics.json`
  - `results/training_log.txt`

**training/config.yaml:**
- Complete hyperparameter configuration (as specified above)

**training/callbacks.py:**
- `EarlyStopping` class
- `ModelCheckpoint` class
- `LearningRateScheduler` class

**training/evaluate.py:**
- `evaluate_model()` - Full evaluation on test set
- Generate confusion matrix
- Per-class metrics
- Save results to results/
- Command: `python training/evaluate.py --model checkpoints/best_model.pth`

**training/README.md:**
- Training instructions
- Hyperparameter tuning guide
- Expected training time
- GPU requirements
- Troubleshooting guide

### inference/

**inference/infer.py:**
- Single image/video inference
- `load_model(checkpoint_path)` - Load trained model
- `predict_image(image_path, model)` - Predict from image
- `predict_batch(image_paths, model)` - Batch prediction
- Command: `python inference/infer.py --model checkpoints/best_model.pth --input image.jpg`

**inference/realtime_demo.py:**
- Real-time webcam demo
- MediaPipe hand detection
- Live prediction with smoothing
- Display prediction + confidence
- FPS counter
- Command: `python inference/realtime_demo.py --model checkpoints/best_model.pth`
- Features:
  - 5-frame prediction smoothing
  - Confidence threshold filtering
  - Text accumulation
  - Keyboard controls (space, c, q)

**inference/tts.py:**
- Text-to-speech module using pyttsx3
- `speak(text)` - Convert text to speech
- Language support (English, Malayalam if available)

**inference/utils.py:**
- `load_class_mapping()` - Load class idx to name mapping
- `preprocess_image(image)` - Preprocessing for inference
- `postprocess_prediction(output)` - Softmax + argmax

**inference/README.md:**
- Inference usage instructions
- Real-time demo guide
- API documentation
- Example outputs

### notebooks/

**notebooks/01_data_exploration.ipynb:**
- Dataset statistics visualization
- Class distribution plots
- Sample visualization (hand landmarks)
- Sequence length analysis for dynamic signs

**notebooks/02_model_experiments.ipynb:**
- Model architecture experiments
- Hyperparameter tuning results
- Ablation studies (dropout, layers, etc.)

**notebooks/03_results_visualization.ipynb:**
- Training curves (loss, accuracy)
- Confusion matrix visualization
- Per-class performance analysis
- Error analysis (misclassified samples)

**notebooks/README.md:**
- Notebook descriptions
- Key findings summary

### results/

**All result files generated by training/evaluate.py:**

**results/metrics.json:**
```json
{
  "test_accuracy": 98.41,
  "test_loss": 0.0623,
  "precision_macro": 0.911,
  "recall_macro": 0.937,
  "f1_macro": 0.920,
  "per_class_metrics": {
    "class_0": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    ...
  },
  "training_time_minutes": 12.5,
  "total_epochs": 22,
  "best_epoch": 15
}
```

**results/confusion_matrix.png:**
- 40×40 confusion matrix heatmap
- Generated using seaborn
- High resolution (300 DPI)

**results/loss_curves.png:**
- Training vs validation loss over epochs
- Mark best epoch
- Show test loss as horizontal line

**results/accuracy_curves.png:**
- Training vs validation accuracy over epochs
- Mark best epoch
- Show test accuracy as horizontal line

**results/per_class_performance.csv:**
```csv
class_idx,class_name,precision,recall,f1_score,support
0,Malayalam_അ,1.00,1.00,1.00,150
...
```

**results/training_log.txt:**
- Complete epoch-by-epoch log
- Format: Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Time

**results/sample_outputs/**
- 10 sample predictions with visualizations
- predictions.txt with details

### checkpoints/

**checkpoints/README.md:**
- Model download links (if >100MB)
- Checkpoint format description
- How to load checkpoints

### docs/

**docs/architecture_diagram.png:**
- Visual diagram of BiLSTM architecture
- Layer dimensions labeled
- Generated using matplotlib

**docs/system_pipeline.png:**
- End-to-end system flow diagram
- Preprocessing → Training → Inference pipeline
- Generated using matplotlib

**docs/technical_report.pdf:**
- Complete technical documentation
- Sections:
  1. Introduction
  2. Dataset Description
  3. Preprocessing Pipeline
  4. Model Architecture
  5. Training Procedure
  6. Results and Evaluation
  7. Conclusion
- Include all graphs and metrics

**docs/dataset_preprocessing.md:**
- Detailed preprocessing documentation
- MediaPipe configuration
- Augmentation techniques
- Quality checks

**docs/training_details.md:**
- Training hyperparameters
- Optimization strategy
- Convergence analysis
- Hardware requirements

### tests/

**tests/test_preprocessing.py:**
- Unit tests for preprocessing functions
- Test MediaPipe extraction
- Test augmentation functions
- Test data validation

**tests/test_model.py:**
- Model architecture tests
- Forward pass tests
- Shape validation tests

**tests/test_inference.py:**
- Inference pipeline tests
- Prediction format validation

### scripts/

**scripts/setup_environment.sh:**
```bash
#!/bin/bash
# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Environment setup complete!"
```

**scripts/run_all.sh:**
```bash
#!/bin/bash
# Complete pipeline: preprocess → train → evaluate
python preprocessing/preprocess.py
python training/train.py
python training/evaluate.py
echo "Pipeline complete!"
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **For classes 7-14:** Generate exactly 50-100 samples per class using ALL augmentation techniques
2. **MediaPipe settings:** Use `min_detection_confidence=0.3` for better recall on problematic Malayalam images
3. **Class weights:** Calculate inverse frequency after augmentation to balance loss
4. **Sequence handling:** Use packed sequences in LSTM for efficiency
5. **Checkpointing:** Save every 5 epochs + best + final
6. **Early stopping:** Patience=7, monitor validation loss
7. **Gradient clipping:** max_norm=1.0 to prevent exploding gradients
8. **Data loading:** Extract to local disk, not from Drive (45x speedup)
9. **Batch size:** 64 for 8GB VRAM GPUs
10. **Real-time demo:** 5-frame smoothing, 0.7 confidence threshold

---

## EXPECTED OUTPUTS AFTER RUNNING

**After preprocessing:**
```
data/processed/
├── train_split.csv (67,805 samples)
├── val_split.csv (13,756 samples)
├── test_split.csv (14,531 samples)
├── class_mapping.csv
└── features/
    ├── class_0_sample_0.npy
    ├── class_0_sample_1.npy
    └── ... (96,092 files)
```

**After training:**
```
checkpoints/
├── best_model.pth (epoch 15, 98.42% val acc)
├── final_model.pth (epoch 22)
├── epoch_5.pth
├── epoch_10.pth
└── epoch_15.pth

results/
├── metrics.json
├── confusion_matrix.png
├── loss_curves.png
├── accuracy_curves.png
├── per_class_performance.csv
├── training_log.txt
└── sample_outputs/
```

**Test Accuracy Targets:**
- Overall: 98%+
- ISL classes (15-39): 98-99%
- Malayalam static (0-6): 97-99%
- Malayalam dynamic (7-14): 80-90% (up from 0%)

---

## VALIDATION CRITERIA

The generated code must:
1. ✅ Run without errors on provided dataset structure
2. ✅ Generate all required files in correct locations
3. ✅ Achieve 98%+ test accuracy overall
4. ✅ Achieve 80%+ accuracy on classes 7-14 (with augmentation)
5. ✅ Complete training in <30 minutes on modern GPU
6. ✅ Support real-time inference (>20 FPS)
7. ✅ Include proper error handling and logging
8. ✅ Follow PEP 8 style guidelines
9. ✅ Include docstrings for all functions/classes
10. ✅ Generate all visualizations and reports

---

**Generate complete, production-ready code for all files following this specification exactly. Every file should be executable and functional, not a template. Include proper imports, error handling, and documentation throughout.**