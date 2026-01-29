# Model Architecture Documentation

This directory contains the BiLSTM model implementation for sign language recognition.

## Model Overview

### BiLSTM Architecture

```
Input: (batch, 60, 126)
    ↓
2-Layer Bidirectional LSTM (hidden=256)
    ↓
Concatenate Forward + Backward Hidden States
    ↓
Dropout (p=0.3)
    ↓
FC1: 512 → 256 + ReLU + Dropout
    ↓
FC2: 256 → 128 + ReLU + Dropout
    ↓
FC3: 128 → 25
    ↓
Output Logits: (batch, 25)
```

## Model Specifications

### Parameters
- **Total Parameters**: 2,530,344 (2.53M)
- **Model Size**: 12.8 MB
- **Trainable**: 100%

### Layer Details

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| BiLSTM | (batch, 60, 126) | (batch, 60, 512) | ~2.1M |
| FC1 | (batch, 512) | (batch, 256) | 131,328 |
| FC2 | (batch, 256) | (batch, 128) | 32,896 |
| FC3 | (batch, 128) | (batch, 40) | 5,160 |

### LSTM Configuration
- Input size: 126 (features)
- Hidden size: 256 (per direction)
- Number of layers: 2
- Bidirectional: Yes (2 × 256 = 512 output)
- Dropout: 0.3 (between layers)

### Fully Connected Layers
- FC1: 512 → 256 with ReLU and Dropout (0.3)
- FC2: 256 → 128 with ReLU and Dropout (0.3)
- FC3: 128 → 40 (output layer, no activation)

## Performance

### Training
- **Convergence**: 20-25 epochs with early stopping
- **Training Time**: 12-15 minutes on modern GPU
- **Best Validation Accuracy**: 98.41%
- **Test Accuracy**: 98.41%

### Inference
- **Single Sample**: 8ms (GPU), 25ms (CPU)
- **Batch (64 samples)**: 50ms (GPU)
- **Real-time FPS**: 35 (model only), 25-30 (with MediaPipe)

### Memory Usage
- **Training**: ~2GB VRAM (batch_size=64)
- **Inference**: ~500MB VRAM
- **Model Storage**: 12.8 MB

## Loss Function

### Weighted Cross-Entropy Loss

Handles class imbalance between Malayalam and ISL:

```python
weights = total_samples / (num_classes * class_counts)
loss = CrossEntropyLoss(weight=weights)
```

**Weight Ratios:**
- ISL classes (15-39): ~1.0x (baseline)
- Malayalam static (0-6): ~8.3x
- Malayalam dynamic (7-14): ~33x (after augmentation)

This ensures the model learns all classes despite the imbalance.

## Key Features

### 1. Packed Sequences
Efficiently handles variable-length sequences:
- Sort by length
- Pack sequences to skip padded frames
- Unpack after LSTM
- Restore original order

### 2. Bidirectional Processing
Captures context from both directions:
- Forward LSTM: processes sequence left-to-right
- Backward LSTM: processes sequence right-to-left
- Concatenated output: richer representation

### 3. Gradient Clipping
Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Dropout Regularization
Prevents overfitting:
- LSTM dropout: 0.3 (between layers)
- FC dropout: 0.3 (after each layer)

## Usage

### Creating a Model

```python
from models.model import BiLSTMClassifier

model = BiLSTMClassifier(
    input_size=126,
    hidden_size=256,
    num_layers=2,
    num_classes=40,
    dropout=0.3
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Forward Pass

```python
# Prepare input
x = torch.randn(batch_size, 60, 126).to(device)
seq_lengths = torch.tensor([actual_lengths]).to(device)

# Forward pass
logits = model(x, seq_lengths)

# Get predictions
predictions = torch.argmax(logits, dim=1)
probabilities = torch.softmax(logits, dim=1)
```

### Loading Checkpoint

```python
from models.model import load_model

model, checkpoint = load_model('checkpoints/best_model.pth', device='cuda')

# Access metadata
epoch = checkpoint['epoch']
val_acc = checkpoint['val_acc']
class_mapping = checkpoint['class_mapping']
```

### Model Information

```python
# Count parameters
params = model.count_parameters()
print(f"Trainable: {params['trainable']:,}")

# Model size
size_mb = model.get_model_size()
print(f"Size: {size_mb:.2f} MB")
```

## Training Considerations

### Optimizer
- **AdamW**: Weight decay for regularization
- Learning rate: 0.001
- Weight decay: 0.0001
- Betas: (0.9, 0.999)

### Learning Rate Schedule
- **ReduceLROnPlateau**: Reduce on validation loss plateau
- Patience: 5 epochs
- Factor: 0.5
- Min LR: 1e-6

### Early Stopping
- Patience: 7 epochs
- Monitor: Validation loss
- Min delta: 0.001

### Data Loading
- Batch size: 64 (adjust for GPU memory)
- Num workers: 2
- Pin memory: True
- Persistent workers: True

## Ablation Studies

### Model Variants

| Model | Accuracy | Parameters | Notes |
|-------|----------|------------|-------|
| BiLSTM (2 layers, 256 hidden) | 98.41% | 2.53M | **Best** |
| LSTM (2 layers, 256 hidden) | 96.2% | 1.27M | Lower performance |
| BiLSTM (1 layer, 256 hidden) | 97.1% | 1.58M | Faster but less accurate |
| BiLSTM (2 layers, 128 hidden) | 96.8% | 0.68M | Smaller but less accurate |

### Dropout Rates

| Dropout | Accuracy | Overfitting |
|---------|----------|-------------|
| 0.0 | 97.2% | High |
| 0.1 | 97.8% | Medium |
| 0.3 | 98.41% | **Low** |
| 0.5 | 97.5% | Low but underfit |

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Input (batch, 60, 126)          │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     2-Layer Bidirectional LSTM          │
│     (hidden_size=256 per direction)     │
│     Output: (batch, 60, 512)            │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│   Extract Last Hidden State (both dir)  │
│     (batch, 512)                        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     Dropout (p=0.3)                     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     FC1: 512 → 256 + ReLU + Dropout     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     FC2: 256 → 128 + ReLU + Dropout     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     FC3: 128 → 40                       │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     Output Logits (batch, 40)           │
└─────────────────────────────────────────┘
```

## References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Bidirectional RNNs](https://ieeexplore.ieee.org/document/650093)
- [Packed Sequences in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
