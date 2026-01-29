# Model Checkpoints

This directory stores trained model checkpoints.

## Checkpoint Files

After training, the following checkpoints will be saved:

### Main Checkpoints
- `best_model.pth` - Model with best validation accuracy
- `final_model.pth` - Model after final epoch
- `epoch_5.pth`, `epoch_10.pth`, etc. - Periodic checkpoints (every 5 epochs)

## Checkpoint Format

Each `.pth` file contains:
```python
{
    'epoch': int,                    # Training epoch number
    'model_state_dict': OrderedDict, # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'train_loss': float,             # Training loss
    'val_loss': float,               # Validation loss
    'train_acc': float,              # Training accuracy
    'val_acc': float,                # Validation accuracy
    'config': dict,                  # Model configuration
    'class_mapping': dict,           # Class idx to name mapping
}
```

## Loading a Checkpoint

```python
import torch
from models.model import BiLSTMClassifier

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')

# Initialize model
model = BiLSTMClassifier(
    input_size=checkpoint['config']['input_size'],
    hidden_size=checkpoint['config']['hidden_size'],
    num_layers=checkpoint['config']['num_layers'],
    num_classes=checkpoint['config']['num_classes'],
    dropout=checkpoint['config']['dropout']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
```

## Model Specifications

### BiLSTM Architecture
- **Parameters**: 2,530,344 (2.53M)
- **Model Size**: 12.8 MB
- **Input Shape**: (batch_size, 60, 126)
- **Output Shape**: (batch_size, 40)

### Expected Performance
- **Test Accuracy**: 98%+
- **Inference Time**: 8ms per sample
- **Real-time FPS**: 35 (model only), 25-30 (with MediaPipe)

## Download Pre-trained Models

If pre-trained models are available (>100MB), download links:
- Best Model: [Download Link]
- Final Model: [Download Link]

## Training Information

To train your own model:
```bash
python training/train.py --config training/config.yaml
```

Expected training time:
- GPU (NVIDIA RTX 3070+): 12-15 minutes
- GPU (NVIDIA GTX 1060): 25-30 minutes
- CPU: 2-3 hours

## Checkpointing Strategy

During training:
1. **Best Model**: Saved whenever validation accuracy improves
2. **Periodic**: Saved every 5 epochs
3. **Final**: Saved after last epoch
4. **Early Stopping**: Training stops if no improvement for 7 epochs

## File Sizes

Typical checkpoint sizes:
- Single checkpoint: ~13 MB
- With optimizer state: ~26 MB
- Full training run (10-12 checkpoints): ~150-200 MB

## Version Compatibility

Checkpoints are compatible with:
- PyTorch >= 2.0.0
- Python >= 3.8
- CUDA >= 11.8 (if GPU-trained)

## Backup Recommendations

For production use:
1. Keep at least 3 best checkpoints
2. Version checkpoints with training dates
3. Store metadata separately (metrics.json)
4. Backup to cloud storage for important models

## Troubleshooting

**Issue: "RuntimeError: Error(s) in loading state_dict"**
- Ensure model architecture matches checkpoint
- Check PyTorch version compatibility

**Issue: "Checkpoint file not found"**
- Run training first: `python training/train.py`
- Check path is correct relative to project root

**Issue: "CUDA out of memory when loading"**
- Load checkpoint to CPU first: `torch.load('model.pth', map_location='cpu')`
- Transfer to GPU if needed: `model.to('cuda')`
