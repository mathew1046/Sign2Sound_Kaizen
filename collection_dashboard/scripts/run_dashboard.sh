#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> INCLUDE-50 Data Collection Dashboard"
echo "Project root: $ROOT"

# Extract reference samples if needed
python3 -c "
from collection_dashboard.server.references import extract_reference_zip
extract_reference_zip()
print('Reference samples ready')
"

# Build frontend
if command -v npm >/dev/null 2>&1; then
  echo "==> Building frontend"
  (cd collection_dashboard/frontend && npm install --silent && npm run build)
else
  echo "WARN: npm not found — skip frontend build (API-only mode)"
fi

# Optional webcam check
python3 -c "
import cv2
cap = cv2.VideoCapture(int(__import__('os').environ.get('CAMERA_INDEX', '0')))
if cap.isOpened():
    print('Webcam OK')
    cap.release()
else:
    print('WARN: Webcam not available — engine will idle until camera connects')
"

export PYTHONPATH="$ROOT:$ROOT/scripts/mspt:${PYTHONPATH:-}"

RTMLIB_LAB="$ROOT/data/include50_rtmlib_1080"
LEGACY_LAB="/media/mathew/OS/Users/augus/INCLUDE_ML/include50_lab"
export INCLUDE_ML_ROOT="${INCLUDE_ML_ROOT:-/media/mathew/OS/Users/augus/INCLUDE_ML}"
export INCLUDE50_VIDEO_ROOT="${INCLUDE50_VIDEO_ROOT:-$INCLUDE_ML_ROOT/include-50}"
export INCLUDE50_VIDEO_MANIFEST_ROOT="${INCLUDE50_VIDEO_MANIFEST_ROOT:-$LEGACY_LAB}"
if [[ -d "$RTMLIB_LAB" ]]; then
  if [[ -z "${INCLUDE50_LAB_ROOT:-}" ]] || [[ "${INCLUDE50_LAB_ROOT}" == "$LEGACY_LAB" ]]; then
    export INCLUDE50_LAB_ROOT="$RTMLIB_LAB"
    echo "Keypoints lab: $INCLUDE50_LAB_ROOT"
    echo "RGB videos:    $INCLUDE50_VIDEO_ROOT"
  fi
fi

# Prefer conda base if system Python lacks FastAPI
if python3 -c "import fastapi" 2>/dev/null; then
  PY=python3
elif command -v conda >/dev/null 2>&1; then
  PY="conda run --no-capture-output -n base python"
else
  echo "ERROR: Install fastapi and uvicorn (see environment.yml)"
  exit 1
fi

echo "==> Starting server on http://localhost:${COLLECTION_PORT:-8010}"
exec $PY -m collection_dashboard.server.main
