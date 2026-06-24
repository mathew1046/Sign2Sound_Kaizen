"""Paths and defaults for the INCLUDE-50 data collection dashboard."""

from __future__ import annotations

import csv
import os
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_ROOT.parent

_DEFAULT_RTMLIB_LAB = PROJECT_ROOT / "data" / "include50_rtmlib_1080"
_DEFAULT_LEGACY_LAB = Path("/media/mathew/OS/Users/augus/INCLUDE_ML/include50_lab")


def _default_include50_lab_root() -> Path:
    if os.environ.get("INCLUDE50_LAB_ROOT"):
        return Path(os.environ["INCLUDE50_LAB_ROOT"])
    if _DEFAULT_RTMLIB_LAB.is_dir():
        return _DEFAULT_RTMLIB_LAB
    return _DEFAULT_LEGACY_LAB


INCLUDE50_LAB_ROOT = _default_include50_lab_root()
INCLUDE_ML_ROOT = Path(
    os.environ.get("INCLUDE_ML_ROOT", "/media/mathew/OS/Users/augus/INCLUDE_ML")
)
INCLUDE50_VIDEO_ROOT = Path(
    os.environ.get("INCLUDE50_VIDEO_ROOT", str(INCLUDE_ML_ROOT / "include-50"))
)


def _default_video_manifest_root() -> Path:
    if os.environ.get("INCLUDE50_VIDEO_MANIFEST_ROOT"):
        return Path(os.environ["INCLUDE50_VIDEO_MANIFEST_ROOT"])
    mounted = INCLUDE_ML_ROOT / "include50_lab"
    if (mounted / "manifests").is_dir():
        return mounted
    return mounted


# RGB source videos for Explore tab (rtmlib manifests only list .npy caches).
INCLUDE50_VIDEO_MANIFEST_ROOT = _default_video_manifest_root()
COLLECTION_OUTPUT_DIR = Path(
    os.environ.get("COLLECTION_OUTPUT_DIR", str(PROJECT_ROOT / "collected_data"))
)
REFERENCE_SAMPLES_DIR = Path(
    os.environ.get("REFERENCE_SAMPLES_DIR", str(PROJECT_ROOT / "reference_samples"))
)
REFERENCE_ZIP = Path(
    os.environ.get("REFERENCE_ZIP", str(PROJECT_ROOT / "include50_word_samples.zip"))
)
WORDS_CSV = Path(os.environ.get("WORDS_CSV", str(PROJECT_ROOT / "include50_words.csv")))
CORPUS_VOCAB_CSV = Path(
    os.environ.get(
        "CORPUS_VOCAB_CSV",
        str(PROJECT_ROOT / "scripts" / "mspt" / "include50_mspt_and_include263_vocabulary.csv"),
    )
)
TRANSCODE_CACHE_DIR = REFERENCE_SAMPLES_DIR / "_transcoded"

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_ENABLED = os.environ.get("CAMERA_ENABLED", "true").lower() in ("1", "true", "yes")
SAMPLES_PER_WORD = int(os.environ.get("SAMPLES_PER_WORD", "10"))
REFERENCE_COUNT = int(os.environ.get("REFERENCE_COUNT", "3"))
AUTO_START = os.environ.get("AUTO_START", "true").lower() in ("1", "true", "yes")

CAM_WIDTH = int(os.environ.get("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.environ.get("CAM_HEIGHT", "960"))
CAM_FPS = float(os.environ.get("CAM_FPS", "10"))

MOTION_THRESHOLD = float(os.environ.get("MOTION_THRESHOLD", "0.008"))
MOTION_START_FRAMES = int(os.environ.get("MOTION_START_FRAMES", "3"))
# Fixed clip length to match INCLUDE-50 (~2–3 s); default 2.5 s
RECORD_DURATION_SEC = float(os.environ.get("RECORD_DURATION_SEC", "2.5"))
COOLDOWN_SEC = float(os.environ.get("COOLDOWN_SEC", "2.0"))
REF_MAX_PLAY_SEC = float(os.environ.get("REF_MAX_PLAY_SEC", "5.0"))
REF_PLAY_COUNTDOWN_SEC = float(os.environ.get("REF_PLAY_COUNTDOWN_SEC", "2.0"))

SERVER_HOST = os.environ.get("COLLECTION_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("COLLECTION_PORT", "8010"))

FRONTEND_DIST = DASHBOARD_ROOT / "frontend" / "dist"


def load_vocabulary() -> list[dict]:
    """Return sorted list of {label_id, word, display_name} for collection (INCLUDE-50)."""
    rows: list[dict] = []
    with WORDS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"].strip()
            rows.append(
                {
                    "label_id": int(row["label_id"]),
                    "word": word,
                    "display_name": word.replace("_", " ").title(),
                }
            )
    rows.sort(key=lambda r: r["label_id"])
    return rows


def load_corpus_vocabulary() -> list[dict]:
    """Return sorted list of {label_id, word, display_name} for full INCLUDE-50+263 corpus."""
    if not CORPUS_VOCAB_CSV.is_file():
        return load_vocabulary()
    rows: list[dict] = []
    extra_id = 50
    with CORPUS_VOCAB_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            word = row["canonical_gloss"].strip()
            in50 = row.get("in_include50_mspt", "").strip().lower() == "yes"
            if in50:
                label_id = int(row["include50_label_id"])
            else:
                label_id = extra_id
                extra_id += 1
            rows.append(
                {
                    "label_id": label_id,
                    "word": word,
                    "display_name": word.replace("_", " ").title(),
                    "in_include50": in50,
                }
            )
    rows.sort(key=lambda r: r["label_id"])
    return rows
