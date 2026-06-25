"""Dashboard paths and defaults — all data lives under the repo."""

from __future__ import annotations

import csv
import os
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_ROOT.parent

# rtmlib MSPT lab (INCLUDE-50 + 263 glosses)
LAB_ROOT = PROJECT_ROOT / "data" / "include50_rtmlib_1080"
WHOLEBODY_DIR = LAB_ROOT / "cache" / "wholebody"
BODY_DIR = LAB_ROOT / "cache" / "body"
FACE_DIR = LAB_ROOT / "cache" / "face"
LEFT_HAND_DIR = LAB_ROOT / "cache" / "left_hand"
META_DIR = LAB_ROOT / "cache" / "meta"
MANIFESTS_DIR = LAB_ROOT / "manifests"

# Legacy names used by skeleton browser / compose
INCLUDE50_LAB_ROOT = LAB_ROOT
SKELETON_DIR = WHOLEBODY_DIR
LANDMARKS_DIR = WHOLEBODY_DIR

VOCAB_CSV = PROJECT_ROOT / "scripts" / "mspt" / "include50_mspt_and_include263_vocabulary.csv"
LABEL_MAP_PATH = DASHBOARD_ROOT / "label_map.json"
MANIFEST_PATH = MANIFESTS_DIR / "all.csv"

CATALOG_V50 = DASHBOARD_ROOT / "catalog_v50.json"
CATALOG_V263 = DASHBOARD_ROOT / "catalog_v263.json"
CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", str(CATALOG_V263)))
ASSETS_DIR = DASHBOARD_ROOT / "assets"
COMPOSE_CACHE_DIR = DASHBOARD_ROOT / "cache" / "compose"
EVALS_DIR = DASHBOARD_ROOT / "evals"

# Data collection
COLLECTION_OUTPUT_DIR = PROJECT_ROOT / "collected_data"
REFERENCE_SAMPLES_DIR = PROJECT_ROOT / "reference_samples"
REFERENCE_ZIP = PROJECT_ROOT / "include50_word_samples.zip"
WORDS_CSV = PROJECT_ROOT / "include50_words.csv"
CORPUS_VOCAB_CSV = VOCAB_CSV
TRANSCODE_CACHE_DIR = REFERENCE_SAMPLES_DIR / "_transcoded"

# RGB source videos — INCLUDE-50 tree on mounted drive (4284 .MOV files)
INCLUDE_ML_ROOT = Path(
    os.environ.get("INCLUDE_ML_ROOT", "/media/mathew/OS/Users/augus/INCLUDE_ML")
)
_MOUNTED_INCLUDE50 = INCLUDE_ML_ROOT / "include-50"
_LOCAL_INCLUDE50 = PROJECT_ROOT / "data" / "include-50"


def get_include50_video_root() -> Path:
    """Resolve INCLUDE-50 RGB tree at call time (mount may appear after server start)."""
    if os.environ.get("INCLUDE50_VIDEO_ROOT"):
        return Path(os.environ["INCLUDE50_VIDEO_ROOT"])
    if _MOUNTED_INCLUDE50.is_dir():
        return _MOUNTED_INCLUDE50
    if _LOCAL_INCLUDE50.is_dir():
        return _LOCAL_INCLUDE50
    return _MOUNTED_INCLUDE50


# Backward-compatible default; prefer get_include50_video_root() in runtime code.
INCLUDE50_VIDEO_ROOT = get_include50_video_root()
INCLUDE50_VIDEO_MANIFEST_ROOT = LAB_ROOT

FPS = 25
FRAME_SIZE = 480
VOCAB_VERSION = 263

CROSSFADE_FRAMES = 10
HOLD_FRAMES = 3
BRIDGE_FRAMES = 10

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GEMMA_MODEL = "gemma-4-26b-a4b-it"

ORIENTATION_REFS_DIR = DASHBOARD_ROOT / "orientation_refs"
ORIENTATION_PROGRESS_PATH = DASHBOARD_ROOT / "orientation_progress.json"
ORIENTATION_FEEDBACK_CACHE_DIR = DASHBOARD_ROOT / "cache" / "orientation_feedback"

# Collection / server
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_ENABLED = os.environ.get("CAMERA_ENABLED", "false").lower() in ("1", "true", "yes")
SAMPLES_PER_WORD = int(os.environ.get("SAMPLES_PER_WORD", "10"))
REFERENCE_COUNT = int(os.environ.get("REFERENCE_COUNT", "3"))
AUTO_START = os.environ.get("AUTO_START", "false").lower() in ("1", "true", "yes")

CAM_WIDTH = int(os.environ.get("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.environ.get("CAM_HEIGHT", "960"))
CAM_FPS = float(os.environ.get("CAM_FPS", "10"))

MOTION_THRESHOLD = float(os.environ.get("MOTION_THRESHOLD", "0.008"))
MOTION_START_FRAMES = int(os.environ.get("MOTION_START_FRAMES", "3"))
RECORD_DURATION_SEC = float(os.environ.get("RECORD_DURATION_SEC", "2.5"))
COOLDOWN_SEC = float(os.environ.get("COOLDOWN_SEC", "2.0"))
REF_MAX_PLAY_SEC = float(os.environ.get("REF_MAX_PLAY_SEC", "5.0"))
REF_PLAY_COUNTDOWN_SEC = float(os.environ.get("REF_PLAY_COUNTDOWN_SEC", "2.0"))

SERVER_HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("DASHBOARD_PORT", "8000"))

FRONTEND_DIST = DASHBOARD_ROOT / "frontend" / "dist"


def ensure_combined_manifest() -> Path:
    """Build manifests/all.csv from train/val/test if missing."""
    if MANIFEST_PATH.is_file():
        return MANIFEST_PATH
    rows: list[str] = []
    header: str | None = None
    for split in ("train", "val", "test"):
        p = MANIFESTS_DIR / f"{split}.csv"
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lines = text.splitlines()
        if header is None:
            header = lines[0]
            rows.append(header)
        for line in lines[1:]:
            rows.append(line)
    if header is None:
        raise FileNotFoundError(f"No manifests under {MANIFESTS_DIR}")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return MANIFEST_PATH


def load_vocabulary() -> list[dict]:
    """Return sorted list of {label_id, word, display_name} for INCLUDE-50 collection."""
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
    """Return sorted list for full 263-gloss corpus."""
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
