"""Read/write collected_data/manifest.json."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.config import (
    COLLECTION_OUTPUT_DIR,
    COOLDOWN_SEC,
    RECORD_DURATION_SEC,
    REF_PLAY_COUNTDOWN_SEC,
    SAMPLES_PER_WORD,
    load_vocabulary,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """Thread-safe manifest persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path or (COLLECTION_OUTPUT_DIR / "manifest.json")
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._ensure_word_settings()
                self._sync_from_disk()
            else:
                self._data = self._fresh_manifest()
                self._sync_from_disk()
                self.save()
            return self._data

    def _fresh_manifest(self) -> dict[str, Any]:
        vocab = load_vocabulary()
        words: dict[str, Any] = {}
        for v in vocab:
            word = v["word"]
            words[word] = {
                "label_id": v["label_id"],
                "display_name": v["display_name"],
                "reference_paths": [],
                "slots": [
                    {"index": i, "status": "empty", "file": None}
                    for i in range(SAMPLES_PER_WORD)
                ],
                "completed_count": 0,
                "cooldown_sec": None,
                "ref_countdown_sec": None,
            }
        return {
            "version": 1,
            "samples_per_word": SAMPLES_PER_WORD,
            "words": words,
            "engine": {
                "state": "idle",
                "current_word": None,
                "current_word_index": 0,
                "current_slot": 0,
                "phase": "idle",
                "ref_index": 0,
                "phase_started_at": None,
                "phase_duration_sec": 0.0,
                "motion": 0.0,
                "total_completed": 0,
                "total_target": len(vocab) * SAMPLES_PER_WORD,
                "paused": False,
                "message": "",
            },
        }

    def _sync_from_disk(self) -> None:
        """Scan collected_data/{word}/ for existing mp4 files."""
        COLLECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for word, entry in self._data["words"].items():
            word_dir = COLLECTION_OUTPUT_DIR / word
            completed = 0
            for slot in entry["slots"]:
                idx = slot["index"]
                fname = f"{word}_{idx:04d}.mp4"
                fpath = word_dir / fname
                if fpath.exists():
                    slot["status"] = "complete"
                    slot["file"] = fname
                    slot.setdefault("recorded_at", _utc_now())
                    completed += 1
                elif slot.get("status") == "complete":
                    slot["status"] = "empty"
                    slot["file"] = None
            entry["completed_count"] = completed
        self._recompute_totals()

    def _ensure_word_settings(self) -> None:
        """Backfill timing fields for manifests created before per-word settings."""
        for entry in self._data.get("words", {}).values():
            entry.setdefault("cooldown_sec", None)
            entry.setdefault("ref_countdown_sec", None)

    def get_word_timing(self, word: str) -> dict[str, float]:
        with self._lock:
            entry = self._data["words"][word]
            cooldown = entry.get("cooldown_sec")
            ref_countdown = entry.get("ref_countdown_sec")
            return {
                "cooldown_sec": COOLDOWN_SEC if cooldown is None else float(cooldown),
                "ref_countdown_sec": (
                    REF_PLAY_COUNTDOWN_SEC if ref_countdown is None else float(ref_countdown)
                ),
            }

    def set_word_timing(
        self,
        word: str,
        *,
        cooldown_sec: float | None = None,
        ref_countdown_sec: float | None = None,
    ) -> dict[str, float]:
        with self._lock:
            if word not in self._data["words"]:
                raise KeyError(word)
            entry = self._data["words"][word]
            if cooldown_sec is not None:
                entry["cooldown_sec"] = max(0.0, min(30.0, float(cooldown_sec)))
            if ref_countdown_sec is not None:
                entry["ref_countdown_sec"] = max(0.0, min(30.0, float(ref_countdown_sec)))
            self.save()
            return self.get_word_timing(word)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def update_engine(self, **kwargs: Any) -> None:
        with self._lock:
            self._data["engine"].update(kwargs)
            self.save()

    def set_references(self, word: str, paths: list[str]) -> None:
        with self._lock:
            self._data["words"][word]["reference_paths"] = paths
            self.save()

    def mark_slot_complete(
        self, word: str, slot_index: int, filename: str, duration_ms: int
    ) -> None:
        with self._lock:
            entry = self._data["words"][word]
            slot = entry["slots"][slot_index]
            slot.update(
                {
                    "status": "complete",
                    "file": filename,
                    "duration_ms": duration_ms,
                    "recorded_at": _utc_now(),
                }
            )
            entry["completed_count"] = sum(
                1 for s in entry["slots"] if s.get("status") == "complete"
            )
            self._recompute_totals()
            self.save()

    def clear_slot(self, word: str, slot_index: int) -> None:
        with self._lock:
            entry = self._data["words"][word]
            slot = entry["slots"][slot_index]
            slot.update({"status": "empty", "file": None, "duration_ms": None, "recorded_at": None})
            entry["completed_count"] = sum(
                1 for s in entry["slots"] if s.get("status") == "complete"
            )
            self._recompute_totals()
            self.save()

    def mark_pending_rerecord(self, word: str, slot_index: int) -> None:
        with self._lock:
            slot = self._data["words"][word]["slots"][slot_index]
            slot["status"] = "pending_rerecord"
            slot["file"] = None
            self._data["words"][word]["completed_count"] = sum(
                1
                for s in self._data["words"][word]["slots"]
                if s.get("status") == "complete"
            )
            self._recompute_totals()
            self.save()

    def _recompute_totals(self) -> None:
        total = 0
        for entry in self._data["words"].values():
            total += entry.get("completed_count", 0)
        self._data["engine"]["total_completed"] = total

    def next_incomplete_slot(self, word: str) -> int | None:
        with self._lock:
            for slot in self._data["words"][word]["slots"]:
                if slot["status"] in ("empty", "pending_rerecord"):
                    return slot["index"]
            return None

    def word_is_complete(self, word: str) -> bool:
        with self._lock:
            return self.next_incomplete_slot(word) is None

    def find_next_word_index(self, start: int = 0) -> int | None:
        vocab = load_vocabulary()
        for i in range(start, len(vocab)):
            if not self.word_is_complete(vocab[i]["word"]):
                return i
        return None

    def slot_path(self, word: str, slot_index: int) -> Path:
        return COLLECTION_OUTPUT_DIR / word / f"{word}_{slot_index:04d}.mp4"

    def reset_all(self) -> None:
        """Delete all collected videos and restore a fresh manifest."""
        with self._lock:
            if COLLECTION_OUTPUT_DIR.exists():
                for entry in COLLECTION_OUTPUT_DIR.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
            self._data = self._fresh_manifest()
            self.save()
