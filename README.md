# Team Kaizen - Sign Language Recognition System

**BiLSTM-based Sign Language Recognition for Indian Sign Language (ISL)**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

A production-ready sign language recognition system achieving **98%+ test accuracy** using Bidirectional LSTM networks. The system recognizes **25 Indian Sign Language (ISL) classes**: A-Z (excluding R).

## 🏗️ Architecture

### BiLSTM Model Specifications
- **Parameters**: 2.53M (12.8 MB)
- **Architecture**: 2-layer Bidirectional LSTM + 3 FC layers
- **Input**: 126-dimensional feature vectors (2 hands × 21 landmarks × 3 coords)
- **Output**: 25 classes (ISL A-Z excluding R)
- **Inference Speed**: 8ms per sample, 35 FPS real-time

```
Input (60, 126) 
    ↓
BiLSTM (hidden=256, layers=2)
    ↓
Dropout (0.3)
    ↓
FC1 (512 → 256) + ReLU + Dropout
    ↓
FC2 (256 → 128) + ReLU + Dropout
    ↓
FC3 (128 → 25)
    ↓
Output (25 classes)
```

## 📊 Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 98.41% |
| Precision (Macro) | 91.1% |
| Recall (Macro) | 93.7% |
| F1-Score (Macro) | 92.0% |
| Training Time | ~12 minutes |
| Real-time FPS | 25-30 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd SIGN2SOUND_Kaizen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the datasets:
   - Malayalam dataset (Static + Dynamic)
   - ISL dataset (.npy files)

2. Organize the data structure:
```
datasets/
├── MALAYALAM/
│   ├── Static/
│   │   ├── Character_1/
│   │   └── ...
│   ├── Dynamic/
│   │   ├── Character_1/
│   │   └── ...
│   └── annotations.csv
└── ISL/
    └── data/
        ├── A/
        ├── B/
        └── ...
```

### Preprocessing

```bash
# Extract features and create train/val/test splits
python preprocessing/preprocess.py \
    --malayalam_path /path/to/MALAYALAM \
    --isl_path /path/to/ISL \
    --output data/processed
```

This will:
- Extract MediaPipe hand landmarks from images
- Apply augmentation (horizontal flip) to increase dataset size
- Create stratified 70/15/15 train/val/test splits
- Generate processed feature files for 25 ISL classes

### Training

```bash
# Train the BiLSTM model
python training/train.py --config training/config.yaml

# Monitor training (outputs saved to results/)
```

Expected training time: 12-15 minutes on modern GPU

### Evaluation

```bash
# Evaluate on test set
python training/evaluate.py --model checkpoints/best_model.pth

# Generates:
# - Confusion matrix (results/confusion_matrix.png)
# - Per-class metrics (results/per_class_performance.csv)
# - Training curves (results/loss_curves.png, results/accuracy_curves.png)
```

### Inference

**Single Image Prediction:**
```bash
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input path/to/image.jpg
```

**Real-time Webcam Demo:**
```bash
python inference/realtime_demo.py --model checkpoints/best_model.pth
```

Controls:
- **Space**: Capture prediction and add to sentence
- **C**: Clear accumulated text
- **Q**: Quit

## 📁 Repository Structure

```
SIGN2SOUND_Kaizen/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── processed/          # Processed features and splits
│   ├── README.md
│   └── statistics.txt
│
├── preprocessing/
│   ├── preprocess.py       # Main preprocessing pipeline
│   ├── augmentation.py     # Data augmentation techniques
│   ├── extract_features.py # MediaPipe feature extraction
│   └── README.md
│
├── features/
│   ├── hand_landmarks.py   # MediaPipe hand detection
│   ├── feature_utils.py    # Padding, masking utilities
│   ├── pose_estimation.py  # Placeholder for future use
│   ├── facial_features.py  # Placeholder for future use
│   └── README.md
│
├── models/
│   ├── model.py           # BiLSTM architecture
│   ├── loss.py            # Weighted loss functions
│   ├── custom_layers.py   # Custom layer implementations
│   └── README.md
│
├── training/
│   ├── train.py           # Training pipeline
│   ├── config.yaml        # Hyperparameters
│   ├── callbacks.py       # Early stopping, checkpointing
│   ├── evaluate.py        # Model evaluation
│   └── README.md
│
├── inference/
│   ├── infer.py           # Single/batch inference
│   ├── realtime_demo.py   # Webcam demo
│   ├── tts.py             # Text-to-speech
│   ├── utils.py           # Inference utilities
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   ├── 03_results_visualization.ipynb
│   └── README.md
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_inference.py
│
├── scripts/
│   ├── setup_environment.sh
│   └── run_all.sh
│
├── checkpoints/
│   └── README.md
│
├── results/
│   └── sample_outputs/
│
└── docs/
    ├── architecture_diagram.png
    ├── system_pipeline.png
    ├── dataset_preprocessing.md
    └── training_details.md
```

## 🔬 Key Features

### Preprocessing Pipeline
- **MediaPipe Hands**: Extract 126-dimensional features (21 landmarks × 2 hands × 3 coords)
- **Aggressive Augmentation**: 6 techniques applied to problematic classes 7-14
  - Gaussian noise (σ=0.01-0.03)
  - Scale variation (0.85-1.15)
  - Translation (±0.1)
  - Rotation (±15°)
  - Horizontal flip
  - Temporal speed variation (0.8-1.2x)

### Training Features
- **Weighted Cross-Entropy Loss**: Handles class imbalance
- **Packed Sequences**: Efficient variable-length handling
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Early Stopping**: Patience=7, monitors validation loss
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Checkpointing**: Saves best and periodic models

### Real-time Inference
- **5-frame Prediction Smoothing**: Reduces jitter
- **Confidence Thresholding**: 0.7 minimum confidence
- **Text-to-Speech**: Converts predictions to audio
- **Performance**: 25-30 FPS with MediaPipe optimization

## 📈 Results

### Overall Performance
- **Test Accuracy**: 98.41%
- **Convergence**: 22 epochs (best at epoch 15)
- **Training Time**: 12.5 minutes on NVIDIA GPU

### Per-Category Accuracy
- **ISL Classes (15-39)**: 98-99%
- **Malayalam Static (0-6)**: 97-99%
- **Malayalam Dynamic (7-14)**: 80-90% (improved from 0% with augmentation)

## 🛠️ Technical Details

### Hardware Requirements
- **Training**: NVIDIA GPU with 8GB+ VRAM (or CPU with longer training time)
- **Inference**: CPU-compatible (GPU recommended for real-time)
- **Storage**: ~2GB for processed data + models

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- MediaPipe 0.10+
- OpenCV 4.5+
- See [requirements.txt](requirements.txt) for full list

## 🐛 Troubleshooting

**Issue: MediaPipe fails to detect hands**
- Solution: Adjust `min_detection_confidence` in config (default: 0.3)

**Issue: Out of memory during training**
- Solution: Reduce batch_size in `training/config.yaml` (try 32 or 16)

**Issue: Low accuracy on classes 7-14**
- Solution: Increase augmentation samples (50 → 100 per class)

**Issue: Slow real-time inference**
- Solution: Ensure GPU is available, reduce frame resolution

## 📚 Citation

If you use this work, please cite:

```bibtex
@software{kaizen_sign2sound_2026,
  author = {Team Kaizen},
  title = {BiLSTM-based Sign Language Recognition for Malayalam and ISL},
  year = {2026},
  url = {<repository-url>}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Team Kaizen

- Developed for sign language accessibility
- Focus on Malayalam and Indian Sign Language
- Target: Real-time, production-ready system

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check [docs/](docs/) for detailed documentation
- Review [notebooks/](notebooks/) for examples

---

**Status**: ✅ Production Ready | **Accuracy**: 98.41% | **Real-time**: 30 FPS
