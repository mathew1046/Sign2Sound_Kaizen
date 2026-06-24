"""Resolve or download MediaPipe task models for MSPT live / preprocess pipelines."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MSPT_PKG = Path(__file__).resolve().parent
PROJECT_ROOT = MSPT_PKG.parent
SCRIPTS_MSPT = PROJECT_ROOT / "scripts" / "mspt"
for p in (SCRIPTS_MSPT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import slr_common as C  # noqa: E402

LOCAL_MODELS = PROJECT_ROOT / "weights" / "mediapipe"

MODEL_FILES = {
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def _has_all_models(d: Path) -> bool:
    return all((d / name).is_file() for name in MODEL_FILES)


def _candidate_dirs() -> list[Path]:
    env = os.environ.get("MEDIAPIPE_MODELS_DIR") or os.environ.get("INCLUDE_ML_MODELS_DIR")
    root = Path(os.environ.get("INCLUDE_ML_ROOT", str(C.INCLUDE_ML_ROOT)))
    dirs: list[Path] = []
    if env:
        dirs.append(Path(env))
    dirs.extend([
        LOCAL_MODELS,
        root / "models",
        C.INCLUDE_ML_ROOT / "models",
        Path("/home/mathew/Downloads/INCLUDE_ML/models"),
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def ensure_models_dir(models_dir: Path | None = None, download: bool = True) -> Path:
    """Return a directory containing all three MediaPipe .task files."""
    if models_dir is not None:
        target = Path(models_dir)
        if _has_all_models(target):
            return target
        if not download:
            return target
    else:
        for d in _candidate_dirs():
            if _has_all_models(d):
                return d
        target = LOCAL_MODELS

    target.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_FILES.items():
        path = target / name
        if path.is_file():
            continue
        if not download:
            continue
        print(f"[mediapipe] downloading {name} -> {path}")
        urllib.request.urlretrieve(url, path)
    missing = [n for n in MODEL_FILES if not (target / n).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MediaPipe model(s):\n"
            + "\n".join(f"  - {target / n}" for n in missing)
            + "\nMount INCLUDE_ML drive or set MEDIAPIPE_MODELS_DIR, or allow auto-download."
        )
    return target


def resolve_models_dir(models_dir: Path | None = None) -> Path:
    return ensure_models_dir(models_dir, download=True)
