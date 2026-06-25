# ISL Alphabet Transformer

Real-time Indian Sign Language alphabet recognition (A–Z) using a PyTorch Transformer encoder on MediaPipe hand landmarks.

## Layout

| File | Purpose |
|------|---------|
| `model.py` | `SignTransformer` architecture with positional encoding |
| `train.py` | Train on `.pose` class folders |
| `inference.py` | Webcam inference with TTS |
| `convert_pose.py` | Convert raw `.npy` landmark data to `.pose` format |
| `record_r.py` | Webcam recorder for the missing ISL `R` class |
| `weights/sign_transformer_alphabet.pth` | Trained weights (26 classes) |

## Quick start

```bash
cd alphabet_transformer

# Real-time inference (space to speak, q to quit)
python inference.py

# Train (requires .pose dataset on disk, not included in repo)
python train.py --data-dir /path/to/ISL_pose

# Convert raw numpy landmarks to .pose
python convert_pose.py --input-dir /path/to/ISL/data --output-dir /path/to/ISL_pose
```

## Dependencies

- PyTorch
- MediaPipe (Tasks API)
- OpenCV
- `pose-format`
- `pyttsx3`

Input shape: 30 frames × 126 features (42 hand landmarks × 3 coordinates).
