# 🤟 GloveTalk: AI-Powered Sign Language Translation System

> **A real-time assistive communication system that translates sign language into speech using Multi-Stream Pose Transformers, Computer Vision, and Embedded Wearable Hardware.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![ESP32](https://img.shields.io/badge/ESP32-Embedded-green.svg)
![RTMLib](https://img.shields.io/badge/RTMLib-WholeBody%20Pose-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

---

## 📖 Overview

GloveTalk is an AI-powered real-time sign language translation system developed by **Team Kaizen** for the IEEE SPS **Sign2Sound** competition.

The system combines computer vision, wearable embedded hardware, and deep learning to recognize sign language gestures and convert them into natural speech. Unlike traditional vision-only or glove-only approaches, GloveTalk adopts a **hybrid architecture**, combining wearable sensors with AI-based pose estimation to improve robustness and usability.

---

## ✨ Features

- 🧠 Multi-Stream Pose Transformer (MSPT)
- 🤖 RTMLib Whole-Body Pose Estimation
- ✋ Wearable Smart Glove with ESP32
- 📡 Wireless Sensor Communication
- 🎯 263 Sign Language Classes
- 🔊 Real-Time Speech Output
- ⚡ Low-Latency Inference
- 🧩 Modular Hardware & Software Architecture

---

# 🏗 System Architecture

```
                    CAMERA
                       │
                       ▼
            RTMLib Pose Estimation
                       │
                       ▼
      Multi-Stream Pose Transformer
                       │
                       ▼
            Gesture Classification
                       │
                       ▼
            Sentence Formation
                       │
                       ▼
              Text-to-Speech Output
```

### Wearable Hardware

```
 Flex Sensors + BNO086 IMU
            │
            ▼
         ESP32 MCU
            │
      ESP-NOW / USB
            │
            ▼
      AI Inference Engine
```
# 🛠️ Hardware

GloveTalk uses a dual-glove wearable architecture to capture both finger articulation and hand orientation in real time. Each glove functions as an independent sensing unit, with data synchronized wirelessly for AI-based gesture recognition.

## Hardware Components

| Component | Quantity | Purpose |
|-----------|:--------:|---------|
| ESP32 Development Board | 2 | Wireless data acquisition and communication |
| Flex Sensors | 10 | Individual finger bend measurement (5 per glove) |
| BNO08X 9-DOF IMU | 2 | Hand orientation and motion tracking |
| Li-ion Battery | 2 | Portable power supply |
| Wearable Gloves | 2 | Sensor mounting platform |
| Jumper Wires & Connectors | As required | Hardware interconnections |

---

## Hardware Architecture

```
          LEFT GLOVE                     RIGHT GLOVE
     ┌────────────────┐            ┌────────────────┐
     │ 5 Flex Sensors │            │ 5 Flex Sensors │
     │   BNO08X IMU   │            │   BNO08X IMU   │
     │     ESP32      │ ─ESP-NOW→  │     ESP32      │
     └────────────────┘            └────────────────┘
                                             │
                                             │ USB/UART
                                             ▼
                                   AI Inference (Python)
                                             │
                                             ▼
                                    Text-to-Speech Output
```

---

## Sensor Configuration

Each glove captures:

- **5 Flex Sensor values** (Thumb, Index, Middle, Ring, Little finger)
- **Quaternion orientation** from the BNO08X IMU
- **Real-time synchronized sensor stream** via ESP-NOW

This produces **18 sensor features per frame**, providing comprehensive information about both finger posture and hand orientation.

---
# 🧠 AI Models

The wearable system employs a hybrid machine learning architecture, selecting the most suitable model based on the type of sign being recognized.

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Random Forest Classifier** | Static sign recognition (alphabets) | Single-frame sensor features (Flex + BNO086) | Alphabet prediction |
| **Long Short-Term Memory (LSTM)** | Dynamic gesture recognition (words & phrases) | 50-frame temporal sensor sequence | Dynamic sign prediction |

### Model Pipeline

```
Sensor Acquisition
(Flex Sensors + BNO086)
          │
          ▼
Feature Extraction
          │
          ├──────────────► Random Forest
          │                 (Static Signs)
          │
          └──────────────► LSTM
                            (Dynamic Signs)
          │
          ▼
Confidence Filtering
          │
          ▼
Text-to-Speech Output
```

### Model Highlights

- **Random Forest**
  - Lightweight and low-latency
  - Optimized for static alphabet recognition
  - Minimal computational overhead

- **LSTM Neural Network**
  - Processes temporal sequences of **50 consecutive frames**
  - Learns motion patterns rather than individual poses
  - Suitable for dynamic gestures such as words and short phrases

- **Inference Engine**
  - Confidence-based prediction filtering
  - Sliding window inference
  - Real-time speech generation using **pyttsx3**
## Key Hardware Features

- 📡 Dual-ESP32 wireless architecture using **ESP-NOW**
- ✋ Independent tracking of all ten fingers
- 🧭 High-precision **9-DOF orientation sensing** with BNO08X
- ⚡ Low-latency embedded communication
- 🔧 Automatic flex sensor calibration stored in ESP32 Flash Memory
- 📈 Real-time signal smoothing and sensor validation for stable predictions
- 🔋 Portable wearable design for untethered operation
---

# 🧠 AI Pipeline

| Component | Technology |
|-----------|------------|
| Pose Estimation | RTMLib |
| Object Detection | YOLOX-X |
| Whole-Body Pose | RTMW-X |
| Recognition Model | Multi-Stream Pose Transformer |
| Framework | PyTorch |
| Input | 133 Whole-Body Keypoints |
| Output | 263 Sign Classes |

---

# 🔧 Hardware

- ESP32 Development Board
- BNO086 9-DOF IMU
- Flex Sensors
- Li-ion Battery
- Wearable Smart Glove
- USB/UART Communication

---

# 📊 Model Performance

| Metric | Value |
|---------|------:|
| Sign Classes | **263** |
| Validation Accuracy | **94.30%** |
| Model Parameters | **3.55 Million** |
| Pose Keypoints | **133** |
| Inference | Real-Time |

---

# 📂 Repository Structure

```
.
├── checkpoints/
├── collection_dashboard/
├── data/
├── docs/
├── inference/
├── models/
├── mspt/
├── preprocessing/
├── scripts/
├── training/
├── README.md
└── requirements.txt
```

---

# 🚀 Installation

```bash
git clone https://github.com/<your-username>/GloveTalk.git

cd GloveTalk

pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train MSPT Model

```bash
python scripts/mspt/run_mspt.py
```

### Live Inference

```bash
python scripts/mspt/rtmlib_live_mspt.py
```

---

# 📷 Demo

### Hardware Prototype

<img width="1204" height="1600" alt="shaheempic" src="https://github.com/user-attachments/assets/de4c4784-5eb2-4921-9c08-3692b9d3a0d0" />


### Live Translation

*(Add GIF/video here)*

### System Architecture

*(Add architecture image here)*

---

# 🛣 Roadmap

- ✅ Multi-Stream Pose Transformer
- ✅ RTMLib Integration
- ✅ Real-Time Sign Recognition
- ✅ ESP32 Smart Glove
- 🔄 Hall-Effect Sensor Evaluation
- 🔄 TensorFlow Lite Deployment
- 🔄 3D Printed Wearable Enclosure
- 🔄 Sentence-Level Translation
- 🔄 Mobile & Edge AI Deployment

---

# 💡 Future Improvements

- Hall-effect sensor based smart glove
- On-device TensorFlow Lite inference
- 3D printed ergonomic enclosure
- Multilingual speech generation
- Facial expression & emotion recognition
- Context-aware sentence generation
- Edge AI deployment on embedded platforms

---

# 👥 Team Kaizen

- **Alex Thomas**
- **Mathew Joseph**
- **Muhammed Shaheem**
- **Roshan Thankachan**

**Department of Electronics & Communication Engineering**  
**Mar Athanasius College of Engineering (MACE)**

---

# 🏆 Competition

**IEEE Signal Processing Society Kerala Chapter**  
**Sign2Sound 2026 – Top 5 Finalist**

---

# 📄 License

This project is released under the **MIT License**.

---

## ⭐ If you found this project useful, consider giving the repository a star!
