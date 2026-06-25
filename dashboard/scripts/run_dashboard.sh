#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Sign2Sound ISL Dashboard"
echo "Project root: $ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts/mspt:${PYTHONPATH:-}"

LAB="$ROOT/data/include50_rtmlib_1080"
if [[ ! -d "$LAB/cache/wholebody" ]]; then
  echo "ERROR: rtmlib wholebody cache missing at $LAB/cache/wholebody"
  echo "       Preprocessed keypoints must be present under data/include50_rtmlib_1080/"
  exit 1
fi

echo "==> Preparing catalog and reference samples"
python3 -c "
from dashboard.server.catalog import ensure_catalog
from dashboard.server.collection.references import extract_reference_zip
ensure_catalog()
extract_reference_zip()
print('Catalog and references ready')
"

if command -v npm >/dev/null 2>&1; then
  echo "==> Building frontend"
  (cd dashboard/frontend && npm install --silent && npm run build)
else
  echo "WARN: npm not found — skip frontend build (API-only mode)"
fi

if python3 -c "import fastapi" 2>/dev/null; then
  PY=python3
elif command -v conda >/dev/null 2>&1; then
  PY="conda run --no-capture-output -n base python"
else
  echo "ERROR: Install fastapi and uvicorn (see dashboard/requirements.txt)"
  exit 1
fi

PORT="${DASHBOARD_PORT:-8000}"
echo "==> Starting server on http://localhost:$PORT"
echo "    Lab root: $LAB"
exec $PY -m dashboard.server.main
