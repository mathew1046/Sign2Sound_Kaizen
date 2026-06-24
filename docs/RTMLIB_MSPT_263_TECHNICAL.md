# Sign2Sound — RTMLib MSPT-263 Technical Document

**Team Kaizen · Sign Language Recognition · Competition Technical Brief**

| Field | Value |
|-------|-------|
| **Model name** | MSPT-RTMLib-263 (Multi-Stream Pose Transformer) |
| **Checkpoint** | `checkpoints/mspt/mspt_rtmlib_263_best.pt` |
| **Output classes** | 263 isolated sign glosses |
| **Validation accuracy (best)** | **94.30%** (epoch 31, early stopping) |
| **Parameters** | 3,554,823 (~13.6 MB FP32) |
| **Pose backend** | rtmlib — YOLOX-X + RTMW-X (COCO-WholeBody 133 keypoints) |
| **Repository path** | `~/Arrakis/Sign2Sound_Kaizen/` |

---

## 1. Executive Summary

Sign2Sound uses a **Multi-Stream Pose Transformer (MSPT)** that classifies isolated sign language clips from **three synchronized pose streams**: hands, upper body, and face. Unlike the original Kaizen BiLSTM pipeline (25 ISL letters, MediaPipe hand landmarks only), the competition model:

1. Extracts **133 whole-body keypoints** per frame using **rtmlib** (YOLOX-X person detection + RTMW-X pose estimation).
2. Normalizes and splits keypoints into hand / body / face tensors.
3. Encodes each stream with a **Transformer encoder**, fuses them with **multi-stream cross-attention (MCA)**, and predicts one of **263 gloss labels**.

The vocabulary covers the **INCLUDE-50 MSPT subset** (50 competition words) plus **213 additional glosses** from the broader INCLUDE corpus (INCLUDE-263), enabling richer pre-training and generalization.

---

## 2. Vocabulary & Dataset

### 2.1 Class structure

| Split | Clips | Purpose |
|-------|-------|---------|
| Train | 3,498 | Model training (+ 8× stochastic augmentation) |
| Validation | 386 | Early stopping & checkpoint selection |
| Test | 374 | Held-out evaluation |
| **Total** | **4,258** | All preprocessed rtmlib clips |

- **263 classes** total (`data/include50_rtmlib_1080/lab_summary.json`)
- **50 classes** are the INCLUDE-50 MSPT competition words (`include50_words.csv`, `label_id` 0–49)
- **213 classes** are extended INCLUDE glosses (`label_id` 50–262)
- Full gloss list: `scripts/mspt/include50_mspt_and_include263_vocabulary.csv`

### 2.2 Source video properties

From preprocessing metadata (`data/include50_rtmlib_1080/metrics.json`):

| Property | Value |
|----------|-------|
| Resolution | Native 1080p (median 1920×1080), no resize |
| Frame rate | Native (all frames retained) |
| Keypoint schema | COCO-WholeBody 133 |
| Hand detection rate | 100% of processed frames |
| Total frames (corpus) | 213,781 |

---

## 3. Preprocessing Pipeline (RTMLib)

Each `.MOV` clip from the INCLUDE corpus is processed offline into per-stream `.npy` caches under `data/include50_rtmlib_1080/cache/`.

### 3.1 Pose extraction stack

| Component | Model | Input size |
|-----------|-------|------------|
| Person detection | YOLOX-X (`yolox_x_8xb8-300e_humanart`) | 640×640 |
| Whole-body pose | RTMW-X SimCC (`rtmw-x_384x288`) | 288×384 |
| Runtime | ONNX Runtime GPU 1.20.2 | CUDA when available |

Implementation: `mspt/rtmlib_preprocess.py` → class `RtmlibWholebodyExtractor`

### 3.2 Keypoint layout (133 → 3 streams)

```
COCO-WholeBody 133 keypoints per frame
├── body      indices  0–16  (17 COCO body joints)  → padded to 33 for MSPT
├── foot      indices 17–22  (stored, not used in MSPT)
├── face      indices 23–90  (68 iBUG face landmarks) → padded to 72 for MSPT
├── left_hand indices 91–111 (21 hand joints)
└── right_hand indices 112–132 (21 hand joints)
```

Each keypoint is stored as `(x_norm, y_norm, confidence, valid)` in `[0, 1]` normalized image coordinates.

### 3.3 Per-stream normalization

Applied at load time (`mspt/rtmlib_io.py`, `mspt/normalize.py`):

| Stream | Joints | Normalization |
|--------|--------|---------------|
| **Hands** | 42 (21L + 21R) | Wrist anchor, scale by wrist→middle-finger-tip distance |
| **Body** | 17 → pad 33 | Hip-midpoint anchor, shoulder-width scale |
| **Face** | 68 → pad 72 | Nose anchor, inter-eye distance scale |

Long clips are **uniformly subsampled** to `max_seq_len = 96` frames.

### 3.4 Preprocessing flowchart

```mermaid
flowchart TD
    A[INCLUDE corpus .MOV clip] --> B[rtmlib Wholebody extractor]
    B --> C[YOLOX-X person detection]
    C --> D[RTMW-X 133-keypoint pose]
    D --> E[Pick best person instance]
    E --> F[Normalize xy + confidence per frame]
    F --> G{Split streams}
    G --> H[left_hand/*.npy]
    G --> I[right_hand/*.npy]
    G --> J[body/*.npy]
    G --> K[face/*.npy]
    G --> L[wholebody/*.npy]
    H --> M[Train/val/test manifests]
    I --> M
    J --> M
    K --> M
    M --> N[MSPT training / inference]
```

---

## 4. Model Architecture — MSPT

**MSPT** (Multi-Stream Pose Transformer) is defined in `mspt/model.py`. It processes three temporal keypoint streams in parallel, fuses them, and outputs a single gloss prediction per clip.

### 4.1 Input tensor shapes

After flattening normalized `(x, y)` coordinates:

| Stream | Shape per timestep | Flat dim | Description |
|--------|-------------------|----------|-------------|
| Hand | `(42, 2)` | **84** | Both hands |
| Body | `(33, 2)` | **66** | COCO-17 padded |
| Face | `(72, 2)` | **144** | iBUG-68 padded |
| Mask | `(T,)` | — | 1 = valid frame, 0 = pad |

Batch input after collation: `(B, T, dim)` where `T ≤ 96`.

### 4.2 Architecture diagram

```mermaid
flowchart TB
    subgraph inputs [Input streams per clip]
        H[Hand sequence B×T×84]
        B[Body sequence B×T×66]
        F[Face sequence B×T×144]
        M[Padding mask B×T]
    end

    subgraph encoders [Stream encoders — identical structure]
        direction TB
        SE1[StreamEncoder hand]
        SE2[StreamEncoder body]
        SE3[StreamEncoder face]
    end

    subgraph stream_enc_detail [Each StreamEncoder]
        direction TB
        LP[Linear in_dim → d_model=128]
        CLS[Prepend learnable CLS token]
        PE[Sinusoidal positional encoding]
        TE[2× TransformerEncoderLayer<br/>4 heads, FFN=256, GELU]
        OUT[Output: CLS embedding 128-d]
    end

    H --> SE1
    B --> SE2
    F --> SE3
    M --> SE1 & SE2 & SE3

    SE1 --> CLS_H[cls_hand 128-d]
    SE2 --> CLS_B[cls_body 128-d]
    SE3 --> CLS_F[cls_face 128-d]

    subgraph fusion [MCAFusion]
        direction TB
        ATT[Multi-head attention:<br/>hand queries body+face context]
        NR[Residual + LayerNorm]
        GATE[Gated concat → 384-d]
    end

    CLS_H --> ATT
    CLS_B --> ATT
    CLS_F --> ATT
    ATT --> NR --> GATE

    subgraph joint [Joint transformer]
        direction TB
        JCLS[Prepend joint CLS token]
        JL[2× TransformerEncoderLayer<br/>d=384, 8 heads, FFN=768]
        JOUT[Joint CLS output 384-d]
    end

    GATE --> JCLS --> JL --> JOUT

    subgraph head [Classification head]
        LN[LayerNorm 384]
        FC[Linear 384 → 263]
        SOFT[Softmax → predicted gloss]
    end

    JOUT --> LN --> FC --> SOFT
```

### 4.3 Component specifications

| Module | Configuration |
|--------|---------------|
| **StreamEncoder** (×3) | `d_model=128`, `nhead=4`, `num_layers=2`, `dim_feedforward=256`, dropout=0.1, GELU |
| **Positional encoding** | Sinusoidal, max length 128 (+1 CLS) |
| **MCAFusion** | 4-head cross-attention (hand → body+face), LayerNorm residual, sigmoid gate on 384-d concat |
| **Joint encoder** | 2 layers, `d_model=384`, `nhead=8`, `dim_feedforward=768`, dropout=0.3 |
| **Classifier** | LayerNorm → Linear(384, 263) |
| **Memory optimizations** | Sequential stream encoding, gradient checkpointing (training), AMP FP16 |

### 4.4 Design rationale

- **Multi-stream**: Sign language depends on hand shape, body posture, and facial expression; separate encoders let each modality develop its own temporal representation before fusion.
- **Hand-centric fusion**: MCA uses the hand CLS token as the query attending to body and face context, reflecting that hand motion is primary for gloss discrimination while body/face provide disambiguating context.
- **CLS-based pooling**: Each stream compresses variable-length sequences into a fixed embedding via a learnable CLS token (ViT-style), avoiding late RNN bottlenecks.
- **263-class head**: Extended vocabulary beyond the 50 competition words improves representation learning on diverse INCLUDE footage.

---

## 5. Training

### 5.1 Training configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=1e-4, weight decay=1e-3) |
| Scheduler | Cosine annealing (T_max=150) |
| Loss | Cross-entropy with label smoothing 0.05 |
| Micro-batch size | 4 |
| Gradient accumulation | 8 (effective batch = **32**) |
| Max sequence length | 96 frames |
| Augmentation repeat | 8× (stochastic, training only) |
| Max epochs | 150 |
| Early stopping patience | 20 (validation accuracy) |
| Mixed precision | FP16 (CUDA AMP) |
| Gradient clipping | max norm 1.0 |
| Hardware target | ~4 GB VRAM / 8 GB RAM |

Training entry point: `scripts/mspt/run_mspt.py --rtmlib --ckpt-name mspt_rtmlib_263_best.pt --num-classes 263`

### 5.2 Data augmentation (training only)

Applied jointly across all three streams (`mspt/augment.py`):

| Augmentation | Description |
|--------------|-------------|
| Horizontal flip | 50% — mirror x, swap left/right hand blocks and body pairs |
| Gaussian jitter | σ=0.02 on valid keypoints |
| Skeleton scale | Uniform 0.8–1.2× |
| Temporal dropout | 10% frame zeroing |
| Temporal resample | 0.8–1.2× speed (aligned across streams) |
| SPOTER perspective | Weak 2D perspective warp (shift ±0.15, scale ±0.2) |

### 5.3 Training flowchart

```mermaid
flowchart LR
    A[Load rtmlib .npy caches] --> B[Normalize streams]
    B --> C{Training?}
    C -->|Yes| D[Stochastic augmentation]
    C -->|No| E[Deterministic load]
    D --> F[Flatten xy → tensors]
    E --> F
    F --> G[Collate batch<br/>pad to batch max T]
    G --> H[MSPT forward pass]
    H --> I[Cross-entropy loss]
    I --> J[Backward + AdamW]
    J --> K{Val acc improved?}
    K -->|Yes| L[Save checkpoint]
    K -->|No| M[Patience counter++]
    L --> N{Patience exceeded?}
    M --> N
    N -->|No| A
    N -->|Yes| O[Best model: mspt_rtmlib_263_best.pt]
```

### 5.4 Training results

| Metric | Value | Source |
|--------|-------|--------|
| Best validation accuracy | **94.30%** | Checkpoint metadata, epoch 31 |
| Training stopped | Epoch 31 | Early stopping (patience 20) |
| Parameters | 3,554,823 | Model state dict |

> **Note:** Test-split accuracy was not persisted in the saved checkpoint (`test_acc: null`). Re-evaluation can be run with:
> ```bash
> cd ~/Arrakis/Sign2Sound_Kaizen
> export PYTHONPATH=$PWD:$PWD/scripts/mspt
> python scripts/mspt/run_mspt.py --rtmlib --eval-only \
>   --checkpoint checkpoints/mspt/mspt_rtmlib_263_best.pt \
>   --lab-root data/include50_rtmlib_1080 --num-classes 263
> ```

### 5.5 Related checkpoints

| Checkpoint | Classes | Val accuracy | Notes |
|------------|---------|--------------|-------|
| `mspt_rtmlib_263_best.pt` | 263 | **94.30%** | **Latest / production model** |
| `mspt_rtmlib_1080_best.pt` | 50* | 92.00% | rtmlib backend, INCLUDE-50 only |
| `mspt_best.pt` | 50 | 78.29% | MediaPipe backend baseline |

\*1080 checkpoint predates explicit `num_classes` field; trained on rtmlib 1080p INCLUDE-50 layout.

---

## 6. Inference Pipeline

### 6.1 Offline evaluation

```
scripts/mspt/eval_confusion_matrix.py --rtmlib --split test
```

Produces confusion matrix PNG + per-class JSON under `collection_dashboard/evals/` (when generated).

### 6.2 Live inference

Entry point: `scripts/mspt/rtmlib_live_mspt.py`

```bash
cd ~/Arrakis/Sign2Sound_Kaizen
export PYTHONPATH=$PWD:$PWD/scripts/mspt
python scripts/mspt/rtmlib_live_mspt.py \
  --checkpoint checkpoints/mspt/mspt_rtmlib_263_best.pt \
  --lab-root data/include50_rtmlib_1080 \
  --video-url http://localhost:8080/video
```

### 6.3 Live inference flowchart

```mermaid
flowchart TD
    A[Video frame source<br/>webcam / MJPEG URL] --> B[Optional resize for pose<br/>max width 960px]
    B --> C[rtmlib WholebodyExtractor<br/>YOLOX + RTMW per frame]
    C --> D[133-kp wholebody buffer]
    D --> E[Split + normalize streams]
    E --> F[Ring buffer ~2.5 s clip]
    F --> G{Clip boundary?}
    G -->|No| A
    G -->|Yes| H[MSPT-263 forward pass GPU]
    H --> I[Softmax over 263 classes]
    I --> J{Confidence ≥ threshold?}
    J -->|Yes| K[Display predicted gloss]
    J -->|No| L[Display uncertain]
    K --> M[Skeleton overlay panel]
    L --> M
    M --> A
```

Default live settings:

| Parameter | Default |
|-----------|---------|
| Clip duration | 2.5 s |
| Gap between clips | 1.0 s |
| Capture FPS | 10 |
| Min confidence | 0.12 (263-way softmax is naturally diffuse) |
| Prediction hold | 2.0 s on screen |

---

## 7. End-to-End System Overview

```mermaid
flowchart TB
    subgraph offline [Offline — training data preparation]
        V[INCLUDE video corpus] --> R[rtmlib preprocess]
        R --> C[Keypoint caches .npy]
        C --> T[MSPT-263 training]
        T --> CKPT[mspt_rtmlib_263_best.pt]
    end

    subgraph online [Online — competition / demo]
        CAM[Webcam or video feed] --> POSE[rtmlib live pose]
        POSE --> MSPT[MSPT-263 classifier]
        CKPT --> MSPT
        MSPT --> GLOSS[Predicted sign gloss]
        GLOSS --> TTS[Text-to-speech optional]
    end

    subgraph collect [Data collection dashboard]
        DASH[collection_dashboard] --> REC[Record 10 clips × 50 words]
        REC --> FINETUNE[Optional MSPT finetune]
        FINETUNE --> CKPT2[mspt_finetuned.pt]
    end
```

---

## 8. Repository Layout

| Path | Description |
|------|-------------|
| `mspt/model.py` | MSPT architecture |
| `mspt/rtmlib_preprocess.py` | rtmlib extraction & wholebody splitting |
| `mspt/rtmlib_io.py` | Cache loading & normalization |
| `mspt/rtmlib_dataset.py` | PyTorch dataset for rtmlib caches |
| `mspt/augment.py` | Training augmentations |
| `scripts/mspt/run_mspt.py` | Training & evaluation CLI |
| `scripts/mspt/rtmlib_live_mspt.py` | Live GPU inference |
| `scripts/mspt/eval_confusion_matrix.py` | Confusion matrix generation |
| `data/include50_rtmlib_1080/` | Preprocessed lab (caches + manifests) |
| `checkpoints/mspt/mspt_rtmlib_263_best.pt` | **Production checkpoint** |
| `collection_dashboard/` | Web UI for data collection |
| `weights/mediapipe/` | Legacy MediaPipe models (fallback viz) |
| `include50_words.csv` | 50 competition word list |

---

## 9. Dependencies

Core Python packages (`scripts/mspt/requirements-rtmlib.txt`):

```
rtmlib>=0.0.13
opencv-python>=4.8.0
numpy>=1.26.0
onnxruntime-gpu==1.20.2
torch>=2.0
pandas
scikit-learn
```

rtmlib ONNX models auto-download to `~/.cache/rtmlib/hub/checkpoints/` on first run.

---

## 10. Comparison with Kaizen BiLSTM Baseline

| Aspect | Kaizen BiLSTM (original) | MSPT-RTMLib-263 (competition) |
|--------|--------------------------|-------------------------------|
| Task | ISL finger-spelling A–Z | Isolated sign word recognition |
| Classes | 25 | 263 |
| Input features | 126-d (2×21 hand landmarks × 3D) | 294-d total across 3 streams (84+66+144) |
| Pose backend | MediaPipe hands | rtmlib wholebody |
| Architecture | 2-layer BiLSTM + FC | 3-stream Transformer + MCA fusion |
| Parameters | ~2.53M | ~3.55M |
| Reported test accuracy | 98.41% (25 ISL classes) | 94.30% val (263 glosses) |

Both pipelines coexist in the repository: BiLSTM under `models/`, `training/`, `inference/`; MSPT under `mspt/`, `scripts/mspt/`.

---

## 11. Key References

- **rtmlib**: [https://github.com/Tau-J/rtmlib](https://github.com/Tau-J/rtmlib)
- **RTMPose / RTMW**: OpenMMLab MMPose whole-body models
- **INCLUDE dataset**: Isolated sign language clips (INCLUDE-50 / extended corpus)
- **COCO-WholeBody**: 133-keypoint whole-body annotation format

---

*Sign2Sound competition technical brief. Checkpoint: `checkpoints/mspt/mspt_rtmlib_263_best.pt` · Lab data: `data/include50_rtmlib_1080/`*
