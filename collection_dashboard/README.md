# INCLUDE-50 Data Collection Dashboard

Autonomous webcam data collection for MSPT finetuning: 10 clips × 50 words.

## Quick start

```bash
cd /path/to/signbert_unofficial
chmod +x collection_dashboard/scripts/run_dashboard.sh
./collection_dashboard/scripts/run_dashboard.sh
```

Open **http://localhost:8010** — collection starts automatically.

## Manual start

```bash
export PYTHONPATH=/path/to/signbert_unofficial
python3 -c "from collection_dashboard.server.references import extract_reference_zip; extract_reference_zip()"
cd collection_dashboard/frontend && npm install && npm run build
python3 -m collection_dashboard.server.main
```

## Environment

| Variable | Default |
|----------|---------|
| `INCLUDE50_LAB_ROOT` | `notebooks/data/include50_rtmlib_1080` when present (keypoint `.npy` caches) |
| `INCLUDE50_VIDEO_ROOT` | `$INCLUDE_ML_ROOT/include-50` (original `.MOV` RGB videos) |
| `INCLUDE50_VIDEO_MANIFEST_ROOT` | `$INCLUDE_ML_ROOT/include50_lab` (maps clip stems → video paths) |
| `INCLUDE_ML_ROOT` | `/media/mathew/OS/Users/augus/INCLUDE_ML` |
| `COLLECTION_OUTPUT_DIR` | `./collected_data` |
| `CAMERA_INDEX` | `0` |
| `CAMERA_ENABLED` | `true` — set `false` to start without opening the webcam |
| `AUTO_START` | `true` |
| `COLLECTION_PORT` | `8010` |
| `RECORD_DURATION_SEC` | `2.5` (use `2.0` for shorter clips) |
| `COOLDOWN_SEC` | `2.0` (pause between back-to-back clips) |

## Workflow

1. **Reference phase** — 3 sample videos play once per word
2. **Collection phase** — records **10 clips automatically** (2.5 s each, 2 s gap); reference videos **loop continuously** on screen; **no motion detection**
3. **Review tab** — delete, re-record, or download any slot

## Export for finetuning

```bash
python3 collection_dashboard/scripts/export_manifest.py -o collected_data/collected_manifest.csv
```

### MSPT finetuning hook

After collection (uses **conda base**):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate base
export INCLUDE50_LAB_ROOT=/path/to/include50_lab
export INCLUDE_ML_ROOT=/path/to/INCLUDE_ML
export PYTHONPATH=/path/to/signbert_unofficial:/path/to/signbert_unofficial/notebooks

# 1. Export manifest + extract npy caches (un-mirrors webcam, stores under collected_data/cache/)
python collected_data/preprocess_collected.py

# 2. Fine-tune from mspt_best.pt (4 GB VRAM defaults)
python notebooks/run_mspt_finetune.py

# Smoke test (4 clips, 2 epochs):
bash collected_data/run_smoke.sh
```

Caches live in `collected_data/cache/{landmarks,mspt_body,landmarks_face}/` — not in include50_lab.

Autoresearch loop: see `autoresearch_mspt/program.md`.

## Output layout

```
collected_data/
  manifest.json
  bank/bank_0000.mp4 … bank_0009.mp4
  hello/hello_0000.mp4 …
```
