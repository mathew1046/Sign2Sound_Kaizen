# Phase 3: Full INCLUDE (~263 glosses)

## Prerequisites

- Raw videos under `$INCLUDE_ML_ROOT/include-50` (full tree, not only INCLUDE-50 subset)
- Expanded `$INCLUDE_ML_ROOT/saved_models/label_map_full.json` with all gloss slugs (generate from folder names)
- `include50_lab` package available via `PYTHONPATH=$INCLUDE_ML_ROOT`

## Pipeline

```bash
cd ~/Arrakis/Sign2Sound_Kaizen
export INCLUDE_ML_ROOT=/media/mathew/OS/Users/augus/INCLUDE_ML
export PYTHONPATH="$PWD:$INCLUDE_ML_ROOT"
LAB="$INCLUDE_ML_ROOT/include50_lab"

# 1. Manifest for all folders in include-50_dataset.csv
python scripts/build_full_include_catalog.py --write-manifest

# 2. Skeleton + optional landmarks (long-running)
python -m include50_lab.preprocess.skeleton --manifest-dir "$LAB/manifests_full" --out "$LAB/cache/skeleton_full"
python -m include50_lab.preprocess.landmarks --manifest-dir "$LAB/manifests_full" --out "$LAB/cache/landmarks_full"

# 3. Dashboard catalog
python scripts/build_dashboard_catalog.py \
  --manifest "$LAB/manifests_full/all.csv" \
  --skeleton-dir "$LAB/cache/skeleton_full" \
  --out dashboard/catalog_v263.json \
  --vocab-version 263

# 4. Export browser assets
python scripts/export_dashboard_assets.py --catalog dashboard/catalog_v263.json --skeleton-dir "$LAB/cache/skeleton_full"

# 5. Point API at new catalog
export CATALOG_PATH=dashboard/catalog_v263.json
uvicorn dashboard.server.main:app --host 0.0.0.0 --port 8000
```

Set `CATALOG_PATH` in [`dashboard/config.py`](config.py) or via the environment.
