"""Resolve 2-3 reference sample videos per INCLUDE-50 word."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from collection_dashboard.config import (
    INCLUDE50_LAB_ROOT,
    INCLUDE_ML_ROOT,
    REFERENCE_COUNT,
    REFERENCE_SAMPLES_DIR,
    REFERENCE_ZIP,
    load_vocabulary,
)
from collection_dashboard.server.transcode import ensure_mp4, reencode_for_browser, video_duration_sec


def extract_reference_zip() -> None:
    """Extract include50_word_samples.zip once if reference dir is empty."""
    if not REFERENCE_ZIP.exists():
        return
    REFERENCE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    has_videos = any(REFERENCE_SAMPLES_DIR.rglob("*.MOV")) or any(
        REFERENCE_SAMPLES_DIR.rglob("*.mp4")
    )
    if has_videos:
        return
    with zipfile.ZipFile(REFERENCE_ZIP, "r") as zf:
        zf.extractall(REFERENCE_SAMPLES_DIR)


def _refs_from_extracted(word: str) -> list[Path]:
    word_dir = REFERENCE_SAMPLES_DIR / word
    if not word_dir.is_dir():
        return []
    files = sorted(word_dir.glob("*.MOV")) + sorted(word_dir.glob("*.mov"))
    files += sorted(word_dir.glob("*.mp4"))
    return files[:REFERENCE_COUNT]


def _refs_from_manifests(word: str) -> list[Path]:
    manifest_dir = INCLUDE50_LAB_ROOT / "manifests_resolved"
    if not manifest_dir.is_dir():
        return []
    paths: list[Path] = []
    for csv_path in sorted(manifest_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("label") == word:
                    p = Path(row["path"])
                    if p.exists():
                        paths.append(p)
                    if len(paths) >= REFERENCE_COUNT:
                        return paths
    return paths


def _refs_from_include50(word: str) -> list[Path]:
    include_dir = INCLUDE_ML_ROOT / "include-50"
    if not include_dir.is_dir():
        return []
    display = word.replace("_", " ")
    paths: list[Path] = []
    for mov in sorted(include_dir.rglob("*.MOV")):
        parent_name = mov.parent.name.lower()
        if display.lower() in parent_name or word.lower() in parent_name.replace(" ", "_"):
            paths.append(mov)
            if len(paths) >= REFERENCE_COUNT:
                break
    return paths


class ReferenceVideoResolver:
    """Resolve and cache reference videos for each gloss."""

    def __init__(self) -> None:
        extract_reference_zip()
        self._cache: dict[str, list[Path]] = {}

    def resolve(self, word: str) -> list[Path]:
        if word in self._cache:
            return self._cache[word]

        paths = _refs_from_extracted(word)
        if len(paths) < REFERENCE_COUNT:
            for p in _refs_from_manifests(word):
                if p not in paths:
                    paths.append(p)
                if len(paths) >= REFERENCE_COUNT:
                    break
        if len(paths) < REFERENCE_COUNT:
            for p in _refs_from_include50(word):
                if p not in paths:
                    paths.append(p)
                if len(paths) >= REFERENCE_COUNT:
                    break

        self._cache[word] = paths[:REFERENCE_COUNT]
        return self._cache[word]

    def resolve_playable(self, word: str) -> list[Path]:
        return [ensure_mp4(p) for p in self.resolve(word)]

    def warm_all_playable(self) -> None:
        """Pre-transcode every reference clip so browser playback is instant."""
        for v in load_vocabulary():
            try:
                self.resolve_playable(v["word"])
            except Exception:
                pass

    def reference_duration_sec(self, word: str) -> float:
        """Total seconds to show all reference clips for a word."""
        paths = self.resolve(word)
        if not paths:
            return 0.0
        return sum(video_duration_sec(p) for p in paths)

    def resolve_all(self) -> dict[str, list[Path]]:
        return {v["word"]: self.resolve(v["word"]) for v in load_vocabulary()}
