# Technical Summary: Sign2Sound Pipeline

This document outlines the technical aspects and pipeline for the ISL alphabet transformer track.

## 1. Data Collection & Preprocessing
* **`record_r.py`**: Created to supplement the Indian Sign Language (ISL) dataset with recordings for the missing 'R' class. Captures 30-frame sequences as `.npy` arrays using the MediaPipe Hands API.
* **`convert_pose.py`**: Script to standardize the raw ISL dataset into `.pose` binary files using the `pose_format` library.
  * **Dataset Layout Integration**: Handled divergent folder schema in the dataset (i.e., 'R' class files were sequence-level `(30, 42, 3)` arrays sitting flat in the root directory, while standard 'A-Z' classes contained nested sub-directories storing individual localized `(126,)` single-frame files). 
  * **Data Reshaping**: Loaded, chronologically sorted, dynamically reshaped flattening arrays, and packed `.npy` inputs appropriately into standard `(frames, 1, 42, 3)` targets required by `NumPyPoseBody`.

## 2. Model Architecture
* **`model.py`**: Contains the `SignTransformer` model implemented in PyTorch.
  * Processes inputs of dimension `126` (42 endpoints from left and right hands * 3 coordinate axes [x,y,z]).
  * Applies a custom `PositionalEncoding` layer with sinusoidal wave logic to preserve sequence time-steps.
  * Feeds into PyTorch's `TransformerEncoder` logic with multi-head attention.
  * Compresses the output sequence via global average pooling and maps to an output fully-connected layer for continuous sequential mapping over `26` alphabetic sign classes.

## 3. Training Pipeline
* **`train.py`**: Executes the dataset iterations to train the PyTorch model.
  * Utilizes `PoseDataset` to iteratively read the serialized `.pose` formats.
  * Applies **2D Augmentations** using `pose.augment2d` to shift 2D scale, rotation, and shear to increase generalization over spatial variances.
  * **Normalization Fixes**: Traced an critical `NaN` gradient loss bug back to the native `pose_format` normalization utility, which occasionally divided by zero on invariant tracked distances. Substituted this with custom sequence-level zero-mean / unit-variance standardizations operations prior to passing matrices through the neural network.
  * Trained over the 26 target classes with CrossEntropyLoss and an AdamW optimizer, saving to `weights/sign_transformer_alphabet.pth`.

## 4. Real-time Inference
* **`inference.py`**: The endpoint for streaming webcam visual inferences into spoken audio outputs.
  * **MediaPipe Tasks API**: Migrated out of the deprecated `solutions` API to heavily optimize continuous hand landmark processing using `mediapipe.tasks.vision.HandLandmarker`.
  * **State Tracking**: Uses a max-length `deque(maxlen=30)` array to maintain an ongoing memory tracker of the latest 30 sequence frames.
  * Inputs undergo identical Z-score temporal standardization as defined during training.
  * High-confidence transformer inferences (>85% thresholds) map alphabet selections sequentially into a string word buffer.
  * **Real-time Engine**: Displays via OpenCV `cv2.imshow` UI windows and consumes `pyttsx3` text-to-speech to physically synthesize and audibly dictate the signed word sequence when committed by the user.

## System Tech Stack:
- **Computer Vision**: OpenCV (`cv2`), MediaPipe (`HandLandmarker`)
- **Deep Learning**: PyTorch (`torch`, `nn.TransformerEncoder`)
- **Data Packaging**: `pose_format` 
- **Voice Synthesis**: `pyttsx3`
