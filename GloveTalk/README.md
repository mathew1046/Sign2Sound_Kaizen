# GloveTalk

Bi-LSTM sign language recognition for dual ESP32 sensor gloves.

## Setup

Use the **conda base** environment:

```bash
conda activate base
cd GloveTalk
pip install -r requirements.txt   # only if packages are missing
```

## Train models

```bash
conda activate base
python scripts/train.py --task both
```

Options:
- `--task words` or `--task alphabet` — train one model
- `--skip-preprocess` — reuse existing `preprocessed/*.npz`

## Live inference

```bash
conda activate base
python inference/live_translator.py      # dynamic words
python inference/alphabet_translator.py  # static letters
```

## Project layout

```
data/raw/          Raw CSV datasets
preprocessed/      Cleaned data, scalers, train/val tensors
weights/           Trained .keras models and class labels
models/            Bi-LSTM architecture
scripts/           Preprocessing and training pipeline
inference/         Live translators
collection/        Hardware data collection scripts
firmware/          ESP32 glove sketches
config/            training.yaml hyperparameters
```

## Model input

- **Shape:** `(30, 40)` — 30 frames × 40 features
- **Features:** 18 raw sensors + 4 relative-orientation quaternion + 18 time-derivatives
