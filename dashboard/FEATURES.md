# Sign2Sound ISL Dashboard — Feature Reference

Unified web UI for Sign2Sound Kaizen: learn Indian Sign Language (ISL) signs, compose sentences, collect webcam training data, and explore the 263-gloss rtmlib corpus. Everything runs from a single server on one port (default **http://localhost:8000**).

---

## Table of contents

1. [Global features](#global-features)
2. [Learn tab](#learn-tab)
3. [Collect tab](#collect-tab)
4. [Review tab](#review-tab)
5. [Explore tab](#explore-tab)
6. [Overview tab](#overview-tab)
7. [Shared components](#shared-components)
8. [Collection engine](#collection-engine)
9. [API reference](#api-reference)
10. [Data and configuration](#data-and-configuration)

---

## Global features

### Header and navigation

- **Five main tabs**: Learn, Collect, Review, Explore, Overview.
- **Corpus summary pill** (always visible): shows total gloss count and number of skeleton clips in the corpus.
- **Collection controls** (Collect and Review tabs): clip progress (`completed / target`), engine state badge, Start / Pause / Resume / Stop, camera toggle, and **Reset data** (deletes all recorded videos and restarts collection from scratch).

### Recording bar

When collection is active and not paused, a full-width progress bar appears at the top of the screen:

- **SIGN NOW** during the recording phase.
- **WAIT** during the cooldown between clips.
- Shows current word, sample slot (e.g. `3/10`), and elapsed percentage.

### Vocabulary scope

| Scope | Count | Used in |
|-------|-------|---------|
| INCLUDE-50 collection words | 50 | Collect, Review, Overview |
| Full rtmlib corpus | 263 glosses | Learn, Explore |

---

## Learn tab

The Learn tab has two sub-modes: **Practice signs** and **Practice sentences**.

### Practice signs (Orientation Coach)

Browse and practice individual ISL signs with skeleton playback and optional webcam feedback.

**Sign browser (sidebar)**

- Searchable list of all 263 glosses (sorted alphabetically by display name).
- Shows how many glosses have skeleton data (`withData / total`).
- Each entry shows variant count; glosses without data are disabled.
- Filter by gloss slug or display name.

**Reference skeleton player**

- Plays the default exemplar for the selected gloss using rtmlib COCO-WholeBody skeleton frames.
- Adjustable playback speed: 0.5×, 0.75×, 1×, 1.25×.
- Loops continuously.

**Practice capture**

- **Record from webcam**: 3-second countdown, then a 2-second clip is captured automatically.
- **Upload video**: analyze a pre-recorded clip from disk.
- Live webcam preview during countdown and recording.

**Orientation analysis**

- Compares your clip against the reference sign using pose/orientation features.
- Results: **Within tolerance**, **Needs correction**, or **Try again** (unusable).
- Detailed error chips per feature (direction, deviation in degrees, severity).
- Optional **Gemma 4** coaching feedback (toggle; requires API access).
- **Mastered** badge when a gloss is marked mastered in progress tracking.
- **Recent attempts** history (last 5 attempts with timestamp and result).

### Practice sentences

Compose and play back continuous sign sequences from English text.

**Sentence builder**

- Free-text English input (e.g. “Good morning, thank you”).
- **Learn this sentence** converts text → ISL glosses → stitched skeleton timeline.
- Optional **Gemini** translation for better gloss ordering (falls back to rule-based mapping if no API key).
- Gloss trail shows the mapped sequence with display names.
- Unknown words are reported when they cannot be mapped.

**Sentence playback**

- Continuous skeleton animation across all glosses in the sentence.
- Current gloss highlighted during playback (“Now signing …”).
- Adjustable speed: 0.5×, 0.75×, 1×, 1.25×.
- Frame count and segment hints shown below the player.

---

## Collect tab

Autonomous webcam recording for INCLUDE-50 training data: **10 clips × 50 words** (500 clips total).

### Collection workflow

1. Click **Start collection** — enables the webcam and starts the collection engine.
2. For each word, the engine cycles through phases (see [Collection engine](#collection-engine)).
3. Nothing records until you explicitly start collection.

### Main panel

**Phase banner**

- Current phase badge: `idle`, `reference`, `recording`, `cooldown`, etc.
- Status message from the engine.
- Note when the sidebar selection differs from the word the engine is collecting.

**Reference sample video**

- Plays up to 3 reference clips per word (from `include50_word_samples.zip`).
- During the **reference** phase: plays all 3 once (engine-driven).
- During **recording**: loops a single reference while you sign.
- Auto-retries on load failure; manual refresh via phase change.

**Live webcam feed**

- MJPEG snapshot stream updated ~10 fps.
- Camera health polled every 3 seconds.
- Shows current word and slot (`1/10`) with REC indicator during recording.

**Recorded clips panel (inline)**

- Grid of 10 slots for the selected word.
- View, Delete, Retake per slot.
- Inline player when a slot is selected.
- **Open full review** jumps to the Review tab.

### Sidebar (Collect mode)

- Filterable list of all 50 collection words.
- Per-word progress bar (`completed / 10`).
- Completed words marked visually.
- Click a word to inspect it without stopping the engine (engine may still be on a different word).

### Word timing panel

Per-word configurable wait times (saved to manifest):

| Setting | Purpose | Default |
|---------|---------|---------|
| Between clips | Cooldown after each recording (WAIT bar) | 2 s |
| After 3 references | Pause after reference videos before first recording | 2 s |

Presets: None (0 s), 0.5 s, 1 s, Default (2 s). Values 0–30 seconds, 0.5 s steps.

---

## Review tab

Full management UI for collected webcam recordings.

### Toolbar

- Word selector dropdown (shows `display_name` and `completed/10`).
- **Export manifest CSV** — download collection metadata for training pipelines.

### Reference samples

- Grid of all 3 reference videos for the selected word (with video controls).

### Your recordings

- 10-slot grid with status: empty, complete, or pending re-record.
- Per-slot actions:
  - **View** — play in the main review player.
  - **Download** — save the MP4 file.
  - **Delete** — remove the clip (with confirmation).
  - **Retake** / **Record** — queue the slot for re-recording by the collection engine.

### Main review player

- Full-size player for the actively selected sample.

---

## Explore tab

Browse the full 263-gloss rtmlib corpus with RGB video and synchronized skeleton overlays.

### Model diagnostics

- Displays evaluation results when available (from `notebooks/eval_confusion_matrix.py`).
- **Accuracy** on the eval split.
- **Best / worst classes** with per-class accuracy.
- **Worst classes** list (clickable — jumps to that gloss in the sidebar).
- **Check skeleton quality** list — classes flagged for manual skeleton review.
- Confusion matrix thumbnail (click to open full PNG).

### Class browser

- Dropdown or sidebar to select any of 263 glosses.
- Shows clip count; non-INCLUDE-50 glosses labeled `(INCLUDE-263)`.
- Summary: total clips, clips with RGB video, clips with skeleton data.

### Side-by-side viewer

- **RGB video** pane (when source videos are mounted).
- **Skeleton** pane — rtmlib COCO-WholeBody frames synced to video playback.
- Skeleton-only auto-play when RGB is unavailable.
- Split badge (train / val / test) on each clip.

### Clip grid

- All clips for the selected class as selectable tiles.
- Per-tile metadata: frame count, skeleton status, video status, split.

### Sidebar (Explore mode)

- Filterable list of all 263 glosses.
- Shows clip count per gloss (no collection progress bars).

---

## Overview tab

Collection progress dashboard (auto-refreshes every 3 seconds).

### Stat cards

| Metric | Description |
|--------|-------------|
| Clips collected | Total completed clips across all words |
| Words complete | Words with all 10 slots filled |
| Overall progress | Percentage of total target (500) |
| Engine state | Current engine state (idle, collecting, paused, etc.) |

### Progress bar

- Visual bar for overall collection completion.

### Incomplete words list

- Every word that still needs recordings.
- Shows `completed/10` and remaining count per word.

---

## Shared components

### Skeleton player

Canvas-based frame renderer used in Learn tab:

- **URL mode**: loads PNG frames from `/api/signs/{gloss}/{exemplar_id}/frames/{n}`.
- **Timeline mode**: plays base64-encoded frames from composed sentence timelines.
- Play/pause, loop, speed multiplier, frame change callbacks.

### Progress sidebar

Reused in Collect, Review, and Explore tabs with mode-specific behavior:

- **collect**: progress bars and completion status.
- **explore**: clip counts only.

---

## Collection engine

Background state machine that autonomously cycles through words and records clips.

### Phases

| Phase | Behavior |
|-------|----------|
| `idle` | Waiting for start |
| `reference` | Plays 3 reference videos sequentially |
| `recording` | Captures ~2.5 s webcam clip |
| `cooldown` | Waits (per-word configurable) before next slot |
| `complete` | All words finished |
| `paused` | User paused; resumes from saved state |
| `error` | Camera or recording failure |

### Defaults

| Parameter | Default |
|-----------|---------|
| Samples per word | 10 |
| Reference videos per word | 3 |
| Record duration | 2.5 s |
| Cooldown between clips | 2.0 s (per-word override) |
| Reference countdown | 2.0 s after references (per-word override) |
| Camera resolution | 1280×960 @ 10 fps |

### Re-record queue

Slots marked for retake (from Collect or Review) are picked up by the engine on the next pass through that word.

### Output

- Videos saved to `collected_data/{word}/sample_{slot}.mp4`.
- Manifest at `collected_data/manifest.json`.

---

## API reference

### Health and camera

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server and camera health |
| GET | `/api/camera/status` | Camera enabled state |
| POST | `/api/camera/enable` | Open webcam |
| POST | `/api/camera/disable` | Close webcam |
| GET | `/api/stream/live.mjpg` | Live MJPEG stream |
| GET | `/api/stream/snapshot.jpg` | Single snapshot frame |

### Learn and composition

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/vocab` | All glosses with exemplars (Learn tab) |
| GET | `/api/signs/{gloss}` | Sign detail and variants |
| GET | `/api/signs/{gloss}/{id}/meta` | Exemplar metadata |
| GET | `/api/signs/{gloss}/{id}/frames/{n}` | Skeleton frame PNG |
| POST | `/api/translate` | English → ISL glosses |
| POST | `/api/compose` | Gloss list → skeleton timeline |
| POST | `/api/sentence` | Translate + compose in one step |

### Orientation coach

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/orientation/vocab` | Glosses available for coaching |
| GET | `/api/orientation/reference/{gloss}` | Reference orientation data |
| GET | `/api/orientation/progress/{gloss}` | Practice history and mastery |
| POST | `/api/orientation/analyze` | Analyze uploaded practice clip |

### Collection

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/collection/vocab` | Per-word collection progress |
| GET | `/api/engine/status` | Engine phase and state |
| POST | `/api/engine/start` | Start collection |
| POST | `/api/engine/stop` | Stop collection |
| POST | `/api/engine/pause` | Pause collection |
| POST | `/api/engine/resume` | Resume collection |
| POST | `/api/data/reset` | Delete all collected data |
| GET/PATCH | `/api/words/{word}/timing` | Per-word wait times |
| GET | `/api/references/{word}` | Reference video list |
| GET | `/api/references/{word}/{idx}` | Reference video file |
| GET | `/api/collected/{word}` | All slots for a word |
| GET | `/api/collected/{word}/{slot}` | Single recorded clip |
| DELETE | `/api/collected/{word}/{slot}` | Delete a clip |
| POST | `/api/collected/{word}/{slot}/rerecord` | Queue slot for re-record |
| GET | `/api/export/manifest.csv` | Export collection manifest |
| GET | `/api/overview` | Overview tab stats |

### Corpus explore

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/include50/corpus/summary` | Corpus-wide stats |
| GET | `/api/include50/corpus/vocab` | All 263 glosses with clip counts |
| GET | `/api/include50/eval` | Model evaluation summary |
| GET | `/api/include50/eval/confusion-matrix.png` | Confusion matrix image |
| GET | `/api/include50/{word}/clips` | Clips for a gloss |
| GET | `/api/include50/clips/{word}/{stem}/video` | RGB video file |
| GET | `/api/include50/clips/{word}/{stem}/meta` | Clip metadata |
| GET | `/api/include50/clips/{word}/{stem}/skeleton/{n}.png` | Skeleton frame |

---

## Data and configuration

### Key data paths

| Path | Purpose |
|------|---------|
| `data/include50_rtmlib_1080/cache/wholebody/` | rtmlib keypoints (263 glosses) |
| `data/include50_rtmlib_1080/manifests/` | train/val/test manifests |
| `dashboard/catalog_v263.json` | Default exemplar per gloss |
| `dashboard/assets/` | Optional PNG frame cache (faster Learn loading) |
| `include50_word_samples.zip` | Reference videos for collection |
| `collected_data/` | Webcam recordings output |
| `INCLUDE-50 RGB tree` (mounted or local) | Source videos for Explore tab |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_PORT` | `8000` | Server port |
| `DASHBOARD_HOST` | `0.0.0.0` | Bind address |
| `GEMINI_API_KEY` | _(unset)_ | Gemini translation for sentences |
| `CATALOG_PATH` | `dashboard/catalog_v263.json` | Gloss catalog |
| `INCLUDE50_VIDEO_ROOT` | auto-detected | RGB video root for Explore |
| `CAMERA_ENABLED` | `false` | Auto-enable camera on start |
| `AUTO_START` | `false` | Auto-start collection engine |
| `SAMPLES_PER_WORD` | `10` | Clips per word |
| `RECORD_DURATION_SEC` | `2.5` | Recording length |
| `COOLDOWN_SEC` | `2.0` | Default cooldown between clips |

### Optional AI integrations

| Feature | Model | Requires |
|---------|-------|----------|
| Sentence gloss ordering | Gemini 2.0 Flash | `GEMINI_API_KEY` |
| Orientation coaching feedback | Gemma 4 | API access (toggle in UI) |

Without API keys, sentence translation uses rule-based fallback and orientation analysis still runs without LLM feedback text.

---

## Quick start

```bash
cd /path/to/Sign2Sound_Kaizen
chmod +x dashboard/scripts/run_dashboard.sh
./dashboard/scripts/run_dashboard.sh
```

Open **http://localhost:8000**.

For setup details, see [`README.md`](README.md).
