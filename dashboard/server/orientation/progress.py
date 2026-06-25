"""Per-gloss orientation practice progress store."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.config import ORIENTATION_PROGRESS_PATH
from dashboard.server.orientation.schemas import OrientationAttempt, OrientationProgress

MASTERED_STREAK = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrientationProgressStore:
    def __init__(self, path: Path | None = None):
        self.path = path or ORIENTATION_PROGRESS_PATH
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": 1, "glosses": {}}
        self.load()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self.path.is_file():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._data = {"version": 1, "glosses": {}}
            return self._data

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def log_attempt(
        self,
        gloss: str,
        overall_result: str,
        error_count: int,
        feedback_text: str,
    ) -> OrientationProgress:
        with self._lock:
            glosses = self._data.setdefault("glosses", {})
            entry = glosses.setdefault(gloss, {"attempts": []})
            attempt = {
                "timestamp": _utc_now(),
                "overall_result": overall_result,
                "error_count": error_count,
                "feedback_text": feedback_text,
            }
            entry["attempts"].append(attempt)
            # Keep last 50 attempts per gloss
            entry["attempts"] = entry["attempts"][-50:]
            self.save()
            return self.get_progress(gloss)

    def get_progress(self, gloss: str) -> OrientationProgress:
        with self._lock:
            entry = self._data.get("glosses", {}).get(gloss, {"attempts": []})
            attempts_raw = entry.get("attempts", [])
            attempts = [OrientationAttempt(**a) for a in attempts_raw]
            recent = attempts[-MASTERED_STREAK:]
            mastered = (
                len(recent) >= MASTERED_STREAK
                and all(a.overall_result == "pass" for a in recent)
            )
            return OrientationProgress(
                gloss=gloss,
                attempts=attempts,
                mastered=mastered,
                attempt_count=len(attempts),
            )
