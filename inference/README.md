# Inference Module

This directory contains modules for real-time and batch inference using trained models.

## Files

### `infer.py`
Single and batch image inference script.

**Features:**
- Load trained model from checkpoint
- Extract features from images using MediaPipe
- Generate predictions with confidence scores
- Batch processing support
- Save results to JSON

**Usage:**
```bash
# Single image prediction
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input path/to/image.jpg

# Batch prediction on directory
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input path/to/image_directory/

# With confidence threshold
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input image.jpg \
    --confidence_threshold 0.7 \
    --output predictions.json
```

**Output:**
```json
[
  {
    "image": "path/to/image.jpg",
    "class_id": 5,
    "class_name": "Malayalam_ഏ",
    "confidence": 0.9834
  },
  ...
]
```

### `realtime_demo.py`
Real-time webcam demo with prediction smoothing and text accumulation.

**Features:**
- Live webcam video processing
- 5-frame prediction smoothing
- Confidence-based filtering (0.7 default)
- Accumulated text building
- Text-to-speech integration
- Interactive controls

**Usage:**
```bash
# Basic real-time demo
python inference/realtime_demo.py --model checkpoints/best_model.pth

# With custom settings
python inference/realtime_demo.py \
    --model checkpoints/best_model.pth \
    --device cuda \
    --smoothing 5 \
    --confidence 0.7
```

**Controls:**
- **SPACE**: Capture prediction and add to sentence
- **C**: Clear accumulated text
- **Q**: Quit demo

**Performance:**
- Input resolution: 640×480 (optimized)
- Target FPS: 25-30
- Latency: ~50ms (MediaPipe + model)

### `tts.py`
Text-to-speech module for audio output.

**Features:**
- Convert predicted signs to speech
- Adjustable speech rate and volume
- Language support (English, extensible)
- Async and sync modes

**Usage:**
```python
from inference.tts import TextToSpeech

tts = TextToSpeech(rate=150, volume=0.8)
tts.speak("Hello World", wait=True)
tts.close()
```

### `utils.py`
Inference utility functions.

**Functions:**
- `load_class_mapping()` - Load class name mapping
- `preprocess_image()` - Prepare image for inference
- `postprocess_prediction()` - Format model output
- `format_prediction_output()` - Pretty-print predictions
- `save_predictions()` - Save results to JSON
- `load_predictions()` - Load predictions from JSON

## Usage Examples

### Example 1: Single Image Prediction

```bash
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input test_image.jpg
```

Output:
```
Image: test_image.jpg
  Prediction: Malayalam_അ
  Confidence: 98.34%
```

### Example 2: Real-time Demo

```bash
python inference/realtime_demo.py \
    --model checkpoints/best_model.pth \
    --device cuda
```

Then:
1. Show sign to camera
2. Press SPACE when confident in prediction
3. Text accumulates at bottom
4. Press C to clear, Q to quit

### Example 3: Batch Processing

```bash
# Process all images in directory
python inference/infer.py \
    --model checkpoints/best_model.pth \
    --input dataset/images/ \
    --output results/predictions.json \
    --confidence_threshold 0.5
```

### Example 4: Programmatic Usage

```python
from inference.infer import SignLanguagePredictor

predictor = SignLanguagePredictor('checkpoints/best_model.pth')

# Single prediction
class_name, confidence = predictor.predict_image('image.jpg')
print(f"{class_name}: {confidence*100:.1f}%")

# Batch prediction
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for result in results:
    print(f"{result['image']}: {result['class_name']} ({result['confidence']*100:.1f}%)")

predictor.close()
```

## Expected Performance

### Speed
- Single image: 50-100ms (CPU), 10-20ms (GPU)
- Real-time: 25-30 FPS with MediaPipe
- Batch: 5-10ms per image (GPU)

### Accuracy
- Overall: 98%+
- Per-class: 80-99% (depending on class)
- Confidence: 90%+ for top predictions

## Real-time Demo Features

### Prediction Smoothing
- 5-frame history to reduce noise
- Mode-based aggregation
- Confidence calculation from consistency

### Text Accumulation
- Builds sentences from individual sign predictions
- Displays at bottom of video
- Can be cleared or read aloud

### Interactive Controls
- Capture: Add current prediction to text
- Clear: Reset accumulated text
- Quit: Exit demo

## Integration with TTS

```python
from inference.realtime_demo import RealtimeDemoApp

app = RealtimeDemoApp('best_model.pth')
# When space is pressed, sign is added to text
# Can optionally call app.tts.speak() to hear each sign
app.run()
```

## Troubleshooting

### Camera not opening
- Check camera is connected and not in use
- Try camera ID 0, 1, 2, ...
- Verify OpenCV installation

### Slow inference
- Ensure GPU is available and being used
- Reduce input resolution (change in realtime_demo.py)
- Use batch processing instead of single images

### Low confidence
- Move hands fully into frame
- Ensure good lighting
- Slow down sign execution
- Try lower threshold (--confidence 0.5)

### No predictions
- Check "No hands detected" message
- Ensure good hand visibility
- Verify model checkpoint is loaded correctly

## Output Formats

### JSON Predictions
```json
{
  "image": "path/to/image.jpg",
  "class_id": 15,
  "class_name": "ISL_A",
  "confidence": 0.9834
}
```

### Console Output
```
Image: image.jpg
  Prediction: ISL_A
  Confidence: 98.34%
```

## Advanced Features

### Batch Processing with GPUs
Already optimized in `infer.py` using DataLoader and batching.

### Custom Confidence Thresholds
```bash
--confidence_threshold 0.9  # Only report high-confidence predictions
```

### Saving Predictions
```bash
--output predictions.json  # Save results for later analysis
```

## Performance Optimization Tips

1. **GPU Usage**: CUDA will automatically be used if available
2. **Batch Size**: Larger batches are faster but need more memory
3. **Resolution**: Lower input resolution = faster processing
4. **Smoothing Window**: Larger = more stable but slower response

## References

- [MediaPipe Real-time Inference](https://google.github.io/mediapipe/solutions/hands.html#live-stream-processing)
- [PyTorch Inference](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Real-time Performance Optimization](https://pytorch.org/docs/stable/notes/inference_optimization.html)
