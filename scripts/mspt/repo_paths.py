"""Repository paths for MSPT / rtmlib scripts in Sign2Sound_Kaizen."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_MSPT = REPO_ROOT / "scripts" / "mspt"
MSPT_CHECKPOINTS = REPO_ROOT / "checkpoints" / "mspt"
RTMLIB_LAB = REPO_ROOT / "data" / "include50_rtmlib_1080"
WEIGHTS_DIR = REPO_ROOT / "weights"
CORPUS_VOCAB_CSV = SCRIPTS_MSPT / "include50_mspt_and_include263_vocabulary.csv"
