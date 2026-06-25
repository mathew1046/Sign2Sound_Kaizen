# Sign2Sound — System Analysis Report

**Project:** GloveTalk / Sign2Sound (Team Kaizen)  
**Competition:** IEEE SPS Sign2Sound — Indian Sign Language (ISL) to Speech  
**Document type:** Technical analysis for external review  
**Scope:** Software architecture, validation metrics, inference performance, engineering challenges, and planned enhancements

---

## 1. Executive Summary

Sign2Sound is a real-time assistive communication system that translates Indian Sign Language into spoken English. The software stack combines multiple recognition modalities:

- **Vision-based word recognition** using a Multi-Stream Pose Transformer (MSPT) over whole-body pose keypoints extracted by RTMLib (YOLOX person detection + RTMW whole-body pose estimation).
- **Vision-based alphabet (finger-spelling) recognition** using a PyTorch Transformer over MediaPipe hand landmarks, optionally preceded by a dedicated hand-region detector.
- **Wearable glove recognition** using dual ESP32 sensor gloves and a TensorFlow BiLSTM classifier for a complementary subset of signs.
- **Natural-language post-processing** that converts sequences of sign glosses into grammatically correct English sentences via rule-based composition and, when available, the Google Gemini API.
- **Educational and data-collection tooling** delivered through web dashboards for learning, exploration, and corpus expansion.

The production word-recognition path is MSPT-263 (263 isolated sign glosses). Alphabet recognition and glove sensing operate as parallel channels and are unified at runtime through a configurable fusion policy. This report documents how the system was built, how it performs, where it faces limitations, and how those limitations are being addressed.

---

## 2. System Architecture

### 2.1 High-level data flow

```
Camera / video feed
        │
        ▼
Whole-body pose extraction (RTMLib: YOLOX + RTMW, 133 keypoints)
        │
        ▼
Stream normalization → hand (84-d) + body (66-d) + face (144-d)
        │
        ▼
MSPT classifier → isolated sign gloss + confidence
        │
        ▼
Gloss buffer → rule-based / Gemini sentence composition → text-to-speech

Parallel paths:
  • MediaPipe hand landmarks → Alphabet Transformer (A–Z finger-spelling)
  • ESP32 dual gloves → BiLSTM (overlap vocabulary + selected letters)
```

### 2.2 MSPT model architecture

MSPT (Multi-Stream Pose Transformer) is the core word-level classifier. It processes three synchronized temporal streams:

| Stream | Input per frame | Role |
|--------|-----------------|------|
| Hands | 42 joints × (x, y) = 84 dimensions | Primary discriminative signal for handshape and motion |
| Body | 33 padded upper-body joints × (x, y) = 66 dimensions | Posture and spatial context |
| Face | 72 padded facial landmarks × (x, y) = 144 dimensions | Non-manual signals and expression |

Each stream passes through an identical Transformer encoder (128-dimensional embedding, 2 layers, 4 attention heads). Stream representations are fused via multi-stream cross-attention (hand queries body and face context), concatenated with a learned gate, and processed by a joint Transformer encoder (384 dimensions, 2 layers, 8 heads). A linear classification head outputs a softmax distribution over 263 gloss labels.

**Model size:** 3,554,823 parameters (~13.6 MB in FP32).

### 2.3 Alphabet recognition architecture

The alphabet model (`SignTransformer`) is a separate PyTorch Transformer encoder that accepts a sliding window of 30 frames, each frame represented by 126 features (42 hand landmarks × 3 coordinates). Predictions are emitted when softmax confidence exceeds 0.85. An optional YOLO-based hand bounding-box crop (Cansik detector, default confidence threshold 0.2) precedes MediaPipe landmark extraction to improve robustness when the signer occupies a small portion of the frame.

### 2.4 Wearable glove architecture

Each glove carries five flex sensors and a BNO08X nine-axis inertial measurement unit. Two ESP32 microcontrollers transmit synchronized sensor frames wirelessly (ESP-NOW) to a host machine over USB serial at 115200 baud. A BiLSTM network ingests windows of 30 frames × 40 engineered features (raw sensors, orientation quaternions, and temporal derivatives) and classifies into a words vocabulary (22 signs) or a separate alphabet vocabulary (19 letters).

### 2.5 Multi-modal fusion

Runtime fusion is governed by a priority policy:

| Modality | Role | Acceptance criteria |
|----------|------|---------------------|
| MSPT (vision words) | Primary word recognizer | Confidence ≥ 0.12 (263-way softmax is naturally diffuse) |
| Glove (wearable) | Confirmatory / fallback | Must agree with a recent MSPT gloss within 1.0 s, or activate fallback after 3.0 s of MSPT silence; requires 5 consecutive stable predictions, margin ≥ 0.25, and sufficient flex-sensor activity |
| Alphabet (vision letters) | Finger-spelling mode | Confidence ≥ 0.85; user toggles spell mode explicitly |

Overlapping glove and MSPT vocabulary is mapped through a shared lexicon (e.g., *thank you*, *hello*, *happy*, *teacher*, *house*, *good morning*, *big/large*, *I*).

### 2.6 Natural-language post-processing

Recognized glosses are buffered during signing. After a configurable pause (default 4.0 seconds), the gloss sequence is converted to speakable English:

1. **Rule-based composition** handles common ISL→English patterns (e.g., pronoun + adjective → copula insertion: *i* + *happy* → "I am happy.").
2. **Gemini API** (`gemini-2.0-flash`) refines sequences that rules flag as requiring language-model assistance, with strict constraints against hallucinating content not present in the gloss list.
3. **Fallback** joins gloss tokens with spaces when neither rules nor the API are available.

A reverse translation path (English → ordered gloss sequence) is also implemented in the educational dashboard, using Gemini with a closed vocabulary and an offline fuzzy matcher as fallback.

---

## 3. Validation Metrics

All reported metrics use **top-1 accuracy** unless otherwise noted. F1, precision, and recall are documented only for the legacy BiLSTM finger-spelling baseline. Metrics were compiled from checkpoint metadata, held-out evaluation splits, training logs, and post-training evaluation runs across the project codebase and associated development archives.

### 3.1 MSPT word recognition models

| Model | Classes | Pose backend | Split | Accuracy | Notes |
|-------|---------|--------------|-------|----------|-------|
| **MSPT-263 (production)** | 263 | RTMLib (YOLOX-X + RTMW-X), native 1080p | Validation (386 clips) | **94.30%** | Best checkpoint at epoch 31; early stopping patience 20 |
| MSPT-263 (production) | 263 | RTMLib | Test (374 clips) | Not persisted | Re-evaluable from held-out test manifest |
| MSPT-263 (production) | 263 | RTMLib | Full preprocessed corpus (4,258 clips) | **98.97%** | Post-training evaluation on entire lab corpus; not a held-out split |
| **MSPT-50 (RTMLib)** | 50 | RTMLib, 1080p | Validation | **92.00%** | INCLUDE-50 competition subset only |
| **MSPT-50 (MediaPipe)** | 50 | MediaPipe hands + body + face | Train (591 clips) | 95.60% | Original baseline pipeline |
| MSPT-50 (MediaPipe) | 50 | MediaPipe | Validation (129 clips) | **78.29%** | Baseline before RTMLib migration |
| MSPT-50 (MediaPipe) | 50 | MediaPipe | Test (130 clips) | **76.92%** | Per-class evaluation available |
| MSPT-50 (finetuned) | 50 | MediaPipe | Validation | 75.97% | Finetuning experiment; below baseline |
| MSPT-50 (960p MediaPipe) | 50 | MediaPipe GPU @ 960p | Test (130 clips) | **85.38%** | Intermediate resolution experiment |

**Dataset scale (MSPT-263):** 4,258 isolated clips across 263 glosses — train 3,498 / validation 386 / test 374. Preprocessing retained native 1080p resolution and achieved 100% hand-presence rate across 213,781 frames under RTMLib extraction.

**Notable per-class weaknesses (MSPT-50 MediaPipe test split):** *hot* (25%), *cell_phone*, *paint*, *teacher*, *white* (0% on single-sample classes).

**Notable per-class weaknesses (MSPT-263 full-corpus eval):** *extra* (0%, confused with *beautiful*), *big_large* (80.95%, confused with *loud*), *dog* (85.0%, confused with *how_are_you*).

### 3.2 Alphabet (finger-spelling) model

| Model | Classes | Input | Metric | Value | Notes |
|-------|---------|-------|--------|-------|-------|
| SignTransformer (alphabet) | 26 (A–Z) | 30 frames × 126-d hand landmarks | Training accuracy (final epoch) | **~94.6%** | Representative training log; no held-out validation split in training script |
| SignTransformer (alphabet) | 26 | Same | Training accuracy (estimated) | **~95%** | Project development report estimate |
| SignTransformer (alphabet) | 26 | Same | Validation accuracy | Not recorded | Training uses full dataset without stratified hold-out |

### 3.3 Legacy BiLSTM finger-spelling baseline

| Model | Classes | Input | Metric | Value |
|-------|---------|-------|--------|-------|
| BiLSTMClassifier | 25 ISL letters | 60 frames × 126-d MediaPipe hands | Validation / test accuracy | **98.41%** |
| BiLSTMClassifier | 25 | Same | Precision (macro) | 91.1% |
| BiLSTMClassifier | 25 | Same | Recall (macro) | 93.7% |
| BiLSTMClassifier | 25 | Same | F1 (macro) | 92.0% |

This earlier pipeline predates MSPT and serves as a reference for isolated letter recognition on a smaller alphabet.

### 3.4 Wearable glove models

| Model | Vocabulary | Architecture | Reported accuracy |
|-------|------------|--------------|-------------------|
| GloveTalk words BiLSTM | 22 signs | 30 × 40 features, BiLSTM | Not committed to repository |
| GloveTalk alphabet BiLSTM | 19 letters | Same architecture family | Not committed to repository |

Training infrastructure logs validation accuracy per epoch, but final benchmark numbers were not archived in the repository at the time of this report.

### 3.5 Early prototype models (development archive)

These models represent earlier research tracks and are not part of the production fusion pipeline:

| Model | Task | Best metric | Value |
|-------|------|-------------|-------|
| WordTransformer (10-class INCLUDE subset) | Isolated word classification | Validation accuracy | **91.67%** (epoch 40, 80/20 split) |
| Sign2Sound v3 NMT (FactorizedTransformer) | Pose tokens → FSW gloss sequence | CHRF (epoch 28) | **84.98** |
| Sign2Sound v3 NMT | Same | BLEU | 0.00 (short sequences; CHRF used for checkpoint selection) |
| Sign2Sound v3 VQ-VAE | Holistic frame quantization | Not logged | Pre-trained weights loaded; no captured training metrics |

---

## 4. Inference Pipeline

### 4.1 Production word recognition (MSPT-263 live)

The live word-recognition pipeline operates as follows:

1. **Frame acquisition** — Webcam or HTTP MJPEG video stream; optional resize to maximum width 720 px for pose extraction throughput.
2. **Asynchronous pose extraction** — Background worker thread runs RTMLib whole-body inference (YOLOX-X detection at 640×640, RTMW-X pose at 288×384) per frame without blocking the display loop.
3. **Stream normalization** — 133 COCO-WholeBody keypoints are split into hand, body, and face streams and anchor-normalized per frame.
4. **Motion-gated clip buffering** — Frames accumulate while inter-frame hand/body motion exceeds a threshold (default 0.008). Clips are capped at 2.5 seconds with a minimum of 6 motion frames before classification.
5. **Classification** — Buffered clip is uniformly subsampled to at most 96 frames; MSPT forward pass on GPU/CPU produces a 263-way softmax.
6. **Confidence gating** — Predictions below 0.12 confidence are treated as uncertain (appropriate for high-class-count softmax).
7. **Display hold** — Accepted prediction shown for 5.0 seconds; 1.0-second gap before the next clip.
8. **Gloss composition** — Accepted glosses enter a buffer; after 4.0 seconds of signing pause, rule-based or Gemini composition produces an English sentence.
9. **Text-to-speech** — Composed sentence spoken via pyttsx3 (runs on a dedicated thread).

Threading separates frame capture, pose estimation, gloss composition (including async Gemini calls), and speech synthesis so the preview loop remains responsive.

### 4.2 Hybrid multi-modal live fusion

The hybrid runtime orchestrates all three recognizers concurrently:

```
Video feed ──┬── RTMLib + MSPT-263 ──────────► FusionPolicy.on_mspt()
             ├── MediaPipe + SignTransformer ──► FusionPolicy.on_alphabet()  [spell mode]
             └── Serial glove stream + BiLSTM ─► FusionPolicy.on_glove()      [confirm-only]

FusionPolicy decisions ──► GlossComposer ──► TTS
```

User controls: quit, force-flush gloss buffer, toggle spell mode, flush accumulated spelled word.

### 4.3 Alphabet (finger-spelling) live pipeline

1. Webcam frame capture.
2. Optional YOLO hand-region crop (detector confidence threshold 0.2).
3. MediaPipe Hand Landmarker extraction (detection / presence / tracking confidence 0.5).
4. Sliding deque of 30 landmark frames (126 features each).
5. Z-score normalization; Transformer forward pass every ~160 ms.
6. Letter appended to spell buffer when confidence > 0.85; buffer cleared after each accepted letter.
7. User triggers speech of accumulated word (spacebar).

### 4.4 Glove live pipeline

1. ESP32 gloves sample flex and IMU data at ~50 Hz.
2. Host receives frames over USB serial; features engineered per frame.
3. Sliding window of 30 frames fed to BiLSTM.
4. Words model: confidence threshold 0.75, 3 consecutive matching predictions required.
5. Alphabet model: confidence threshold 0.85, 4 consecutive matching predictions required.
6. In fusion mode, glove tokens are confirmatory only unless fallback mode is enabled.

### 4.5 Pseudo-continuous sign recognition (segmentation strategy)

Because no large-scale Indian continuous sign language recognition (CSLR) dataset exists, the project addresses continuous signing through two complementary approaches:

**A. Synthetic continuous sessions (training-time pseudo-CSLR)**  
Isolated landmark clips from the INCLUDE corpus are programmatically stitched into continuous sessions with inter-sign idle frames. This produces training data for Connectionist Temporal Classification (CTC) models and continuous pose datasets, simulating multi-sign utterances without requiring annotated continuous video.

**B. MSPT-as-segmenter (inference-time pseudo-CSLR)**  
At inference, the isolated-sign MSPT classifier is applied as a lightweight segmenter over overlapping temporal windows rather than relying on a dedicated boundary-detection model:

| Parameter | Typical value | Interpretation |
|-----------|---------------|----------------|
| Window duration | 1.5 s | Local context for each classification |
| Stride | 0.3 s | Overlap for temporal continuity |
| High, stable top-1 confidence | Sustained across windows | Likely inside a sign |
| Low confidence or class flipping | Transient | Transition or idle period |
| Peak confidence | Local maximum | Approximate sign centre; boundaries expanded outward until confidence drops |

The current production live deployment uses motion-gated fixed-length clips (2.5 s) as a simpler segmentation heuristic. The sliding-window MSPT segmenter represents the designed evolution toward true continuous recognition without requiring a separately trained boundary model.

---

## 5. Inference Latency

Formal end-to-end latency benchmarks were not instrumented for every component. The following figures are drawn from documented defaults, measured sub-component timings, and architectural constraints.

### 5.1 MSPT word recognition

| Stage | Latency / throughput | Notes |
|-------|---------------------|-------|
| Pose extraction (RTMLib live) | Variable; EMA pose FPS displayed in status bar | Dominated by YOLOX + RTMW per frame; async worker decouples from display |
| Pose extraction (MediaPipe pilot, 640p) | 8.3–11.8 FPS | Offline preprocessing benchmark on 10 pilot clips |
| MSPT forward pass | Not separately benchmarked | ~3.55M parameters; peak VRAM 43–86 MB in related experiments |
| Clip segmentation floor | **~3.5–5.5 s** between predictions | 2.5 s clip + 1.0 s gap + 5.0 s display hold |
| Gloss composition pause | 4.0 s default | Configurable utterance pause before Gemini/rules fire |
| Gemini API call | Runtime-dependent | Measured by smoke-test tooling (min/max/mean/p50/p95); not checked into repository |

**Effective word recognition rate:** Approximately one isolated gloss every 3.5–5.5 seconds under default clip settings, excluding NLP composition and TTS time.

### 5.2 Alphabet recognition

| Stage | Latency | Notes |
|-------|---------|-------|
| Inference interval | **~160 ms** (~6 FPS classification rate) | Hard-coded minimum interval between predictions |
| Sliding window fill | 30 frames at camera rate | ~1 s at 30 FPS before first prediction possible |
| Model inference | Sub-frame (GPU) | Dominated by MediaPipe landmark extraction |

### 5.3 Legacy BiLSTM baseline

| Configuration | Latency |
|---------------|---------|
| Model only (GPU) | 8 ms / sample |
| Model only (CPU) | 25 ms / sample |
| With MediaPipe extraction | ~50 ms / sample (~25–30 FPS end-to-end) |

### 5.4 Wearable glove

| Stage | Latency |
|-------|---------|
| Sensor window | 30 frames at 50 Hz ≈ **0.6 s** of history per prediction |
| Stability gating | 3–5 consecutive matching predictions before emit |
| Effective glove latency | **~0.6–1.5 s** after sign onset, plus fusion agree-window constraints |

### 5.5 End-to-end spoken output

From sign completion to spoken English sentence:

```
Sign motion → clip buffer (≤2.5 s) → MSPT classify → gloss buffer
    → signing pause (4.0 s) → Gemini/rules compose → TTS speak
```

Under default settings, **6–10+ seconds** can elapse from the end of a multi-gloss signing sequence to spoken output, dominated by segmentation, utterance pause, and optional cloud NLP latency.

---

## 6. Engineering Challenges and Solutions

### 6.1 Pose estimation quality under resolution reduction

**Challenge.** The initial word-recognition pipeline relied on MediaPipe for hand, body, and face landmark extraction. When input frames were downsampled for real-time throughput (e.g., to 224 px), hand detection rates degraded substantially. A pilot comparison across 10 INCLUDE clips showed mean any-hand detection of **39.3%** at 224 px versus **54.8%** at 640 px under MediaPipe GPU extraction. Individual clips such as *cell phone* dropped from 92.5% to 1.9% any-hand rate when resolution was reduced. Poor landmark quality propagated directly into MSPT classification errors on visually similar signs.

**Solution.** The production pipeline migrated to **RTMLib** with YOLOX-X person detection and RTMW-X whole-body pose estimation, processing video at **native 1080p** without frame skipping. On the full 4,258-clip preprocessed corpus, RTMLib achieved **100% any-hand frame rate** across 213,781 frames. For alphabet recognition, a dedicated **YOLO hand-region detector** (Cansik, confidence threshold **0.2**) crops the signing hand before MediaPipe landmark extraction, improving robustness when the signer is distant or occupies a small image region. MediaPipe hand landmarker internal thresholds remain at **0.5** for detection, presence, and tracking.

### 6.2 Limited training data for Indian Sign Language

**Challenge.** Public ISL datasets are scarce relative to the vocabulary required for practical communication. Many glosses in the extended INCLUDE corpus have fewer than 20 clips, and some classes contain only a single exemplar.

**Solution.** A multi-pronged data expansion strategy was applied:

- **Stochastic augmentation (8× repeat)** during MSPT training: horizontal flip with left/right joint swapping, Gaussian keypoint jitter (σ = 0.02), skeleton scaling (0.8–1.2×), temporal dropout (10% frames), temporal resampling (0.8–1.2× speed), and SPOTER-style perspective warping.
- **GloveTalk preprocessing augmentation:** raw sensor noise injection, amplitude scaling, feature dropout, time shifting, and sliding-window extraction (30-frame windows, stride 5) with an 8× multiplier for the words model.
- **Alphabet pseudo-sequences:** single-frame sensor snapshots expanded into 30-frame synthetic temporal sequences (30 variants per sample) to compensate for the absence of continuous alphabet recordings from gloves.
- **Vocabulary expansion to 263 classes:** pre-training on 213 extended INCLUDE glosses beyond the 50-word competition subset to improve representation learning and generalization.
- **Web-based collection dashboard:** structured tooling for recording additional clips per gloss to support future fine-tuning.

### 6.3 Absence of facial features in early models

**Challenge.** Early MSPT variants used hand and body streams only. Indian Sign Language conveys substantial meaning through facial expression (mouthings, eyebrow position, affect). Omitting the face stream left discriminative information unused, particularly for adjective and emotional glosses.

**Solution.** A dedicated **face encoder stream** (144 dimensions from 72 padded iBUG facial landmarks) was integrated as a full peer stream in MSPT, encoded by its own Transformer and fused via cross-attention. Face landmarks are extracted from RTMLib COCO-WholeBody indices 23–90, nose-anchored and scaled by inter-eye distance. The three-stream architecture explicitly models the linguistic role of non-manual signals alongside manual features.

### 6.4 No Indian continuous sign language recognition dataset (pseudo-CSLR)

**Challenge.** Continuous Sign Language Recognition (CSLR) — recognizing multiple signs in an unsegmented video stream — requires temporally annotated continuous corpora. No sufficiently large Indian CSLR dataset was available for end-to-end training.

**Solution.** The project developed a **pseudo-CSLR** framework with two layers:

1. **Training:** Isolated INCLUDE clips are stitched into synthetic continuous sessions with random idle frames between signs, producing CTC-trainable sequences and per-frame gloss labels. This is implemented in the continuous pose session builder and supports BiLSTM-CTC and SignBERT+ research tracks.

2. **Inference:** The trained isolated-sign MSPT classifier serves as a **confidence-based segmenter** over overlapping sliding windows (e.g., 1.5 s window, 0.3 s stride). High, stable top-1 confidence indicates sign interior; low confidence or rapid class changes indicate transitions; peak confidence locates sign centres with outward boundary expansion. This avoids training a separate boundary detector while leveraging the strong isolated-sign classifier.

The production live system currently uses motion-gated clip segmentation as an interim deployment; the sliding-window segmenter is the designed path to full continuous recognition.

### 6.5 Lack of grammatical structure in raw gloss output

**Challenge.** ISL gloss sequences follow sign-order conventions that differ from English syntax. Concatenating classifier outputs (e.g., *i · happy · bank*) produces ungrammatical speech unsuitable for assistive communication.

**Solution.** An **intelligent NLP composition layer** (`GlossComposer`) buffers gloss tokens and, after a signing pause, produces natural English:

- **Rule-based matching** handles high-frequency patterns: pronoun–copula insertion (*i* + *happy* → "I am happy."), compound gloss preservation (*good_morning* → "Good morning."), and light reordering.
- **Gemini API integration** (`gemini-2.0-flash`) refines complex sequences under strict system instructions that forbid inventing content beyond the input gloss list. Output is constrained to a single JSON sentence.
- **Fallback join** concatenates glosses when neither rules nor the API are available.
- **Reverse translation** in the educational dashboard maps English sentences to ISL gloss order using the same closed vocabulary.

### 6.6 Integrating vision, hardware, and alphabet modalities

**Challenge.** Three independent recognizers — vision words (MSPT), vision letters (alphabet Transformer), and wearable gloves (BiLSTM) — operate on different input domains, vocabularies, latencies, and confidence scales. Naïve fusion would produce conflicting or duplicate outputs.

**Solution.** A **unified fusion orchestrator** with an explicit policy layer:

- MSPT is the **primary** word-level authority.
- The glove operates in **confirm-only** mode: it reinforces MSPT when both modalities agree within a 1.0-second window, or optionally provides **fallback** output after 3.0 seconds of MSPT silence.
- Alphabet recognition activates only in **spell mode** (user-toggled), appending high-confidence letters to a spell buffer distinct from the gloss composition path.
- A shared **fusion vocabulary map** aligns glove labels with MSPT gloss slugs for the overlapping subset.
- All accepted tokens feed a common gloss composer and TTS output path in the hybrid live runtime.

### 6.7 Additional challenges

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **263-way softmax confidence dilution** | Raw probabilities are low even for correct predictions; a 0.12 threshold is required vs. typical 0.5+ for fewer classes | Confidence threshold tuned per model; per-class calibration planned |
| **Rare-class data scarcity** | Single-exemplar glosses (*extra*, *thin*, *nice*) show 0% or near-0% accuracy | User-collected dataset expansion; collection dashboard for targeted recording |
| **Cloud NLP dependency** | Gemini API adds latency and requires network connectivity | Rule-based path operates fully offline; Gemini is optional (`--no-gemini`) |
| **Pose extraction compute cost** | RTMLib whole-body inference is the live bottleneck | Async pose worker, frame resize to 720 px, configurable detection interval |
| **Domain shift (corpus → webcam)** | INCLUDE studio footage may differ from live signing conditions | Fine-tuning pipeline on user-collected clips; orientation feedback in dashboard |
| **Temporal segmentation in live use** | Fixed clip length may truncate or merge adjacent signs | MSPT-as-segmenter sliding-window approach; motion-gated adaptive buffering |
| **Glove vocabulary coverage** | Only 22 words and 19 letters vs. 263 vision glosses | Vision-primary architecture; glove as robustness layer for overlap set |

---

## 7. Future Enhancements

### 7.1 Planned enhancements (project roadmap)

1. **Unified multi-input model**  
   Consolidate alphabet finger-spelling, glove sensor input, and word-level pose recognition into a single architecture with shared representations and a unified output head. This would eliminate the fusion policy complexity and enable the model to learn cross-modal correlations directly.

2. **User-collected ISL dataset**  
   Build a substantial community- and user-contributed dataset through the collection dashboard, targeting glosses with weak coverage, regional ISL variation, and diverse signing environments. This data will support fine-tuning MSPT-263 and reducing domain shift between corpus and live conditions.

### 7.2 Recommended additional enhancements

| Enhancement | Rationale |
|-------------|-----------|
| **Deploy MSPT sliding-window segmenter in live pipeline** | Replace fixed 2.5 s clips with overlapping-window confidence segmentation for true continuous recognition |
| **On-device / edge deployment** | Quantize MSPT and pose models (INT8) for Raspberry Pi / Jetson inference without cloud dependency |
| **Per-class confidence calibration** | Temperature scaling or Platt scaling on the 263-way head to improve threshold reliability |
| **Offline grammar model** | Train a small on-device seq2seq gloss→English model to remove Gemini latency and connectivity requirements |
| **Active learning loop** | Flag low-confidence live predictions for automatic inclusion in the collection dashboard queue |
| **WER / CSLR evaluation suite** | Adopt Word Error Rate and continuous recognition benchmarks alongside top-1 accuracy for pseudo-CSLR validation |
| **Multi-signer generalization** | Expand training data across signers, ages, and skin tones to reduce signer-specific bias |
| **Glove vision co-calibration** | Time-align glove sensor windows with video frames for joint multimodal training rather than post-hoc fusion |
| **Real-time orientation coaching** | Extend dashboard orientation feedback into the live inference loop to guide signers toward detectable poses |
| **Fine-grained error analysis dashboard** | Persist live confusion matrices and per-gloss failure rates to guide targeted data collection |

---

## 8. Summary Tables

### 8.1 Production model comparison

| Component | Modality | Classes | Best reported accuracy | Production role |
|-----------|----------|---------|------------------------|-----------------|
| MSPT-263 | Vision (pose) | 263 | 94.30% (val) | Primary word recognition |
| SignTransformer | Vision (hands) | 26 | ~95% (train) | Finger-spelling |
| GloveTalk BiLSTM | Wearable | 22 words / 19 letters | Not archived | Confirmatory / fallback |
| GlossComposer | NLP | — | — | Grammar and TTS preparation |
| BiLSTM (legacy) | Vision (hands) | 25 | 98.41% (val) | Superseded reference baseline |

### 8.2 Latency summary

| Path | Dominant latency | Approximate end-to-end |
|------|------------------|------------------------|
| MSPT word (per gloss) | Clip segmentation | 3.5–5.5 s |
| MSPT + NLP + TTS | Utterance pause + API | 6–10+ s |
| Alphabet letter | 30-frame buffer + 160 ms interval | ~1–2 s per letter |
| Glove word | 30-frame sensor window + stability gate | 0.6–1.5 s |
| Legacy BiLSTM | MediaPipe + model | ~50 ms / frame |

---

## 9. Conclusion

Sign2Sound combines state-of-the-art whole-body pose estimation, multi-stream Transformer classification, wearable sensing, and NLP post-processing into a modular assistive communication platform for Indian Sign Language. The migration from MediaPipe to RTMLib at native resolution, integration of facial features, aggressive data augmentation, pseudo-CSLR segmentation, and multi-modal fusion each address specific engineering challenges posed by limited ISL resources and the complexity of real-time sign translation.

The production MSPT-263 model achieves **94.30% validation accuracy** on 263 glosses — a substantial undertaking given vocabulary breadth and data scarcity. Inference latency is currently bounded by pose extraction and clip-based segmentation rather than the classifier itself. The planned unified model, user-collected dataset, and deployment of the MSPT-as-segmenter sliding-window approach represent the most impactful next steps toward a robust, continuous, and deployable ISL translation system.

---

*Report prepared from the Sign2Sound / GloveTalk codebase, technical documentation, checkpoint metadata, evaluation artifacts, and associated development archives. Metrics reflect the state of the project at the time of compilation.*
