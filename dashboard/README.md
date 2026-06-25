# Sign2Sound ISL Dashboard

Unified web UI for Sign2Sound Kaizen: learn signs, compose sentences, collect webcam data, and explore the 263-gloss rtmlib corpus.

**One server, one port** — no external `INCLUDE_ML_ROOT` or env exports required. All keypoint data comes from `data/include50_rtmlib_1080/` in this repo.

## Quick start

```bash
cd /path/to/Sign2Sound_Kaizen
chmod +x dashboard/scripts/run_dashboard.sh
./dashboard/scripts/run_dashboard.sh
```

Open **http://localhost:8000**

## Tabs

| Tab | Description |
|-----|-------------|
| **Learn** | Browse 263 glosses with rtmlib skeleton playback; compose sentences |
| **Collect** | Autonomous webcam recording — 10 clips × 50 INCLUDE-50 words |
| **Review** | Delete, re-record, or download collected clips |
| **Explore** | Browse full corpus clips with skeleton overlay |
| **Overview** | Collection progress summary |

## Setup (manual)

```bash
cd /path/to/Sign2Sound_Kaizen
pip install -r dashboard/requirements.txt

# Build gloss catalog (auto-built on first server start if missing)
python scripts/build_dashboard_catalog.py

# Optional: PNG frame cache for faster Learn tab loading
python scripts/export_dashboard_assets.py

# Frontend
cd dashboard/frontend && npm install && npm run build

# Run
python -m dashboard.server.main
```

## Data paths (all repo-local)

| Path | Purpose |
|------|---------|
| `data/include50_rtmlib_1080/cache/wholebody/` | rtmlib keypoints (263 glosses) |
| `/media/mathew/OS/Users/augus/INCLUDE_ML/include-50/` | RGB source videos for Explore tab (auto-detected when mounted) |
| `data/include50_rtmlib_1080/manifests/` | train/val/test manifests |
| `dashboard/catalog_v263.json` | Default exemplar per gloss |
| `include50_word_samples.zip` | Reference videos for collection (auto-extracted) |
| `collected_data/` | Webcam recordings output |

Optional override: set `INCLUDE50_VIDEO_ROOT` if the mount path differs.

## Optional environment

| Variable | Default |
|----------|---------|
| `DASHBOARD_PORT` | `8000` |
| `GEMINI_API_KEY` | _(unset — rule-based translation fallback)_ |
| `CAMERA_ENABLED` | `false` |
| `AUTO_START` | `false` |
| `CATALOG_PATH` | `dashboard/catalog_v263.json` |

## API highlights

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/vocab` | Learn tab — all glosses with exemplars |
| GET | `/api/signs/{gloss}/{id}/frames/{n}` | Skeleton frame PNG |
| POST | `/api/sentence` | Translate + compose timeline |
| GET | `/api/collection/vocab` | Collection progress per word |
| GET | `/api/include50/corpus/vocab` | Full 263-gloss corpus |
| GET | `/api/include50/clips/{word}/{stem}/skeleton/{n}.png` | Explore skeleton frame |

## Related

- MSPT technical brief: [`../docs/RTMLIB_MSPT_263_TECHNICAL.md`](../docs/RTMLIB_MSPT_263_TECHNICAL.md)
- Data layout: [`../data/README.md`](../data/README.md)
