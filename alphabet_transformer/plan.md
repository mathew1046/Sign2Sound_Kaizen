# GloveTalk v2: Transformer Architecture Implementation Plan

## Phase 1: Data Migration & Standardization
**Objective:** Move away from raw NumPy arrays and standardize the ISL dataset into the `.pose` binary format for native normalization and fast loading.

* **Step 1.1: Record the Missing 'R' Class**
    * Write an OpenCV capture script to record 120 sequences (30 frames each) of the ISL 'R' sign.
    * Run the existing MediaPipe pipeline to extract the (x,y,z) coordinates.
    * Save as `.np` files to match the rest of the dataset.
* **Step 1.2: Convert `.np` to `.pose`**
    * Write a conversion script using the `pose-format` library.
    * Iterate through all 3600 files per alphabet.
    * Map the 42 landmarks (21 left + 21 right) into a `Pose` object.
    * Save all files as `.pose` binaries.

## Phase 2: Building the Transformer Model
**Objective:** Replace the BiLSTM with a PyTorch Transformer Encoder to better capture the spatial and temporal relationships of continuous sign sequences.

* **Step 2.1: Positional Encoding**
    * Implement a positional embedding layer. Transformers don't inherently understand time (sequence order) like LSTMs do, so you must inject positional data into the pose vectors.
* **Step 2.2: The Pose Encoder**
    * Implement a `TransformerEncoder` using PyTorch (`nn.TransformerEncoder`).
    * Recommended starting hyperparameters:
        * `num_layers` = 4 to 6
        * `num_heads` = 4 or 8
        * `hidden_size` = 256
* **Step 2.3: The Classification Head**
    * Add a linear layer + Softmax at the end of the Transformer to map the encoded features to the 26 alphabet classes (plus a "Neutral/Silence" class).

## Phase 3: Training Pipeline & Augmentation
**Objective:** Train the model using robust data augmentation to ensure real-world accuracy matches testing accuracy.

* **Step 3.1: Implement Data Loaders**
    * Create a PyTorch DataLoader that reads the `.pose` files.
* **Step 3.2: Native Normalization**
    * Call `pose.normalize()` on every loaded sequence to automatically translate and scale the coordinates based on shoulder/body width.
* **Step 3.3: Spatial Augmentation**
    * During training, dynamically apply `pose.augment2d(rotation_std=0.2, scale_std=0.2, shear_std=0.2)` to prevent the model from overfitting to the specific camera angles in the dataset.
* **Step 3.4: Train and Validate**
    * Train using Cross-Entropy Loss and an AdamW optimizer. 
    * Monitor validation accuracy to prevent overfitting.

## Phase 4: Real-Time Inference & TTS Integration
**Objective:** Deploy the model on the live webcam feed and hook it up to the speech engine.

* **Step 4.1: The Sliding Window Queue**
    * Initialize a `collections.deque` with a `maxlen=30`.
    * As the live webcam captures frames, extract MediaPipe coordinates, normalize them, and push them to the deque.
* **Step 4.2: Continuous Prediction**
    * Every $N$ frames (e.g., every 5 frames to save CPU), pass the 30-frame deque to the Transformer.
    * If the prediction confidence exceeds a high threshold (e.g., 0.85) and is *not* the "Neutral" class, register the letter.
* **Step 4.3: Text-to-Speech (sign2sound)**
    * Buffer the recognized letters into words.
    * Pass the completed string to your TTS API (Sarvam or AWS Nova) to output the spoken audio.

## Dataset

Provide your own ISL landmark dataset as class folders (`A`–`Z`) of `.pose` or raw `.npy` files. Use `convert_pose.py` and `record_r.py` to prepare data before training.