# Training Pipeline Documentation

This directory contains scripts for training and evaluating the BiLSTM model.

## Files

### `train.py`
Main training script that implements the complete training pipeline.

**Features:**
- Custom dataset loading from preprocessed .npy files
- Training with validation
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience=7)
- Gradient clipping (max_norm=1.0)
- Model checkpointing (best + periodic)
- Training metrics logging and visualization

**Usage:**
```bash
# Basic training
python training/train.py --config training/config.yaml

# With specific device
python training/train.py --config training/config.yaml --device cuda

# Resume from checkpoint
python training/train.py --config training/config.yaml --resume checkpoints/epoch_10.pth
```

**Outputs:**
- `checkpoints/best_model.pth` - Best model based on validation accuracy
- `checkpoints/final_model.pth` - Model after final epoch
- `checkpoints/epoch_*.pth` - Periodic checkpoints (every 5 epochs)
- `results/training_metrics.json` - Training/validation metrics
- `results/training_curves.png` - Loss and accuracy plots

### `callbacks.py`
Callback implementations for training monitoring.

**Classes:**
- `EarlyStopping` - Stop training if no improvement
- `ModelCheckpoint` - Save best and periodic checkpoints
- `LearningRateMonitor` - Track learning rate changes

### `evaluate.py`
Model evaluation script for test set analysis.

**Features:**
- Comprehensive metrics (accuracy, precision, recall, F1)
- Per-class performance analysis
- Confusion matrix visualization
- Classification report

**Usage:**
```bash
# Evaluate best model
python training/evaluate.py --model checkpoints/best_model.pth

# With custom test set
python training/evaluate.py \
    --model checkpoints/best_model.pth \
    --test_csv data/processed/test_split.csv \
    --output_dir results/evaluation
```

**Outputs:**
- `results/test_metrics.json` - Overall metrics
- `results/per_class_metrics.csv` - Per-class metrics
- `results/confusion_matrix.png` - Confusion matrix heatmap
- `results/classification_report.txt` - Detailed classification report

### `config.yaml`
Complete hyperparameter configuration file.

**Key Sections:**
- `model`: Architecture specifications
- `training`: Optimizer and batch settings
- `scheduler`: Learning rate scheduling
- `early_stopping`: Early stopping configuration
- `data`: Dataset paths and loading options
- `checkpointing`: Model saving configuration
- `logging`: Output directory and verbosity

## Expected Performance

### Training Convergence
- Convergence: 20-25 epochs with early stopping
- Training time: 12-15 minutes on modern GPU (NVIDIA RTX 3070+)
- Batch size: 64 (for 8GB VRAM)

### Test Accuracy
- **Overall**: 98%+
- **ISL (0-24)**: 98-99%

### Per-Metric Performance
- Precision (macro): 91.1%
- Recall (macro): 93.7%
- F1-Score (macro): 92.0%

## Training Workflow

1. **Preprocessing** (if not done)
   ```bash
   python preprocessing/preprocess.py \
       --isl_path /path/to/ISL \
       --output data/processed
   ```

2. **Training**
   ```bash
   python training/train.py --config training/config.yaml
   ```

3. **Evaluation**
   ```bash
   python training/evaluate.py --model checkpoints/best_model.pth
   ```

4. **Analysis** (Jupyter notebooks)
   ```bash
   jupyter notebook notebooks/03_results_visualization.ipynb
   ```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config.yaml (try 32 or 16)
- Ensure no other GPU processes are running
- Use CPU if GPU memory is insufficient

### Training is too slow
- Verify GPU is being used (check `nvidia-smi`)
- Increase `batch_size` if memory allows
- Use `use_amp: true` for mixed precision (faster on RTX cards)

### Model not converging
- Check class weights are being used
- Verify data loading is correct
- Try different learning rates (0.0005 to 0.01)
- Increase training epochs

### Early stopping too aggressive
- Increase `early_stopping.patience` (default: 7)
- Decrease `early_stopping.min_delta` (default: 0.001)

## Advanced Usage

### Custom Loss Function
Edit `config.yaml`:
```yaml
training:
  loss_type: "focal"  # or "label_smoothing", "ce"
```

### Resuming Training
```bash
python training/train.py \
    --config training/config.yaml \
    --resume checkpoints/epoch_10.pth
```

### Different Device
```bash
python training/train.py \
    --config training/config.yaml \
    --device cpu
```

## Monitoring

### During Training
- Check `results/training_metrics.json` in real-time
- View `results/training_curves.png` after each validation

### Learning Rate Schedule
Enabled by default in config. Reduces LR by 0.5 if validation loss doesn't improve for 5 epochs.

### Early Stopping
Stops training if validation loss doesn't improve for 7 epochs (configurable).

## Output Structure

```
checkpoints/
├── best_model.pth        # Best validation accuracy
├── final_model.pth       # Final model after training
├── epoch_5.pth           # Checkpoint at epoch 5
├── epoch_10.pth          # Checkpoint at epoch 10
└── ...

results/
├── training_metrics.json # Loss/accuracy history
├── training_curves.png   # Loss and accuracy plots
├── test_metrics.json     # Test set metrics
├── per_class_metrics.csv # Per-class performance
├── confusion_matrix.png  # Confusion matrix
└── classification_report.txt
```

## References

- [PyTorch Training Loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
- [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)
