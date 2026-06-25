"""Autonomous collection state machine."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any

import numpy as np

from dashboard.config import (
    COLLECTION_OUTPUT_DIR,
    COOLDOWN_SEC,
    RECORD_DURATION_SEC,
    REF_PLAY_COUNTDOWN_SEC,
    SAMPLES_PER_WORD,
    load_vocabulary,
)
from dashboard.server.collection.transcode import reencode_for_browser, video_duration_sec
from dashboard.server.collection.manifest import ManifestStore
from dashboard.server.collection.references import ReferenceVideoResolver
from dashboard.server.collection.webcam import WebcamCapture


class Phase(str, Enum):
    IDLE = "idle"
    REFERENCE = "reference"
    RECORDING = "recording"
    COOLDOWN = "cooldown"
    COMPLETE = "complete"
    PAUSED = "paused"
    ERROR = "error"


class CollectionEngine:
    """Background thread that cycles words and records clips."""

    def __init__(
        self,
        manifest: ManifestStore,
        webcam: WebcamCapture,
        refs: ReferenceVideoResolver,
    ):
        self.manifest = manifest
        self.webcam = webcam
        self.refs = refs
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()
        self._rerecord_queue: list[tuple[str, int]] = []
        self._vocab = load_vocabulary()
        self._pre_pause: dict[str, Any] | None = None
        self._replay_references_on_resume = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="collection-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def pause(self) -> None:
        snap = self.manifest.snapshot()["engine"]
        self._pre_pause = {
            "state": snap.get("state", "collecting"),
            "phase": snap.get("phase", Phase.IDLE.value),
            "message": snap.get("message", ""),
            "phase_started_at": snap.get("phase_started_at"),
            "phase_duration_sec": snap.get("phase_duration_sec", 0.0),
        }
        self._pause.set()
        self.manifest.update_engine(state="paused", paused=True)

    def resume(self) -> None:
        self._pause.clear()
        self._replay_references_on_resume = True
        snap = self.manifest.snapshot()["engine"]
        word = snap.get("current_word") or ""
        display = word.replace("_", " ") if word else "the sign"
        self.manifest.update_engine(
            paused=False,
            state="collecting",
            phase=Phase.REFERENCE.value,
            ref_index=0,
            message=f"Resuming — watch all 3 references before signing: {display}",
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def queue_rerecord(self, word: str, slot_index: int) -> None:
        with self._lock:
            if (word, slot_index) not in self._rerecord_queue:
                self._rerecord_queue.append((word, slot_index))

    def _wait_if_paused(self) -> bool:
        while self._pause.is_set():
            self.manifest.update_engine(state="paused", paused=True)
            if self._stop.is_set():
                return False
            time.sleep(0.2)
        return True

    def _sleep_interruptible(self, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stop.is_set():
                return False
            if not self._wait_if_paused():
                return False
            if self._replay_references_on_resume:
                return False
            time.sleep(min(0.1, end - time.monotonic()))
        return True

    def _play_references(self, word: str, word_idx: int) -> bool:
        """Play all reference clips once, then a short countdown before signing."""
        self._replay_references_on_resume = False
        ref_paths = self.refs.resolve(word)
        self.manifest.set_references(word, [str(p) for p in ref_paths])

        for ref_i, ref_path in enumerate(ref_paths):
            if self._stop.is_set() or not self._wait_if_paused():
                return False
            duration = video_duration_sec(str(ref_path))
            wait_sec = max(duration + 0.35, 1.0)
            self._update_engine(
                current_word=word,
                current_word_index=word_idx,
                phase=Phase.REFERENCE.value,
                ref_index=ref_i,
                phase_duration_sec=wait_sec,
                motion=0.0,
                message=f"Watch reference {ref_i + 1}/{len(ref_paths)}: {word.replace('_', ' ')}",
            )
            if not self._sleep_interruptible(wait_sec):
                if self._replay_references_on_resume:
                    return self._play_references(word, word_idx)
                return False

        if ref_paths:
            countdown = self.manifest.get_word_timing(word)["ref_countdown_sec"]
            if countdown > 0 and not self._sleep_interruptible(countdown):
                if self._replay_references_on_resume:
                    return self._play_references(word, word_idx)
                return False
        return True

    def _update_engine(self, **kwargs) -> None:
        if "phase" in kwargs and "phase_started_at" not in kwargs:
            kwargs["phase_started_at"] = time.time()
        eng = self.manifest.snapshot()["engine"]
        eng.update(kwargs)
        self.manifest.update_engine(**eng)

    def _run(self) -> None:
        if not self.webcam.ensure_reader():
            self._update_engine(
                state="error",
                phase=Phase.ERROR.value,
                message=self.webcam.error or "Camera unavailable",
            )
            return
        time.sleep(0.3)

        word_idx = self.manifest.find_next_word_index(0)
        if word_idx is None:
            self._update_engine(state="complete", phase=Phase.COMPLETE.value, message="All 500 clips collected")
            return

        self._update_engine(state="collecting", message="Collecting automatically…")

        while not self._stop.is_set():
            if not self._wait_if_paused():
                break

            with self._lock:
                if self._rerecord_queue:
                    word, slot = self._rerecord_queue.pop(0)
                    word_idx = next(i for i, v in enumerate(self._vocab) if v["word"] == word)
                    if not self._collect_word(word, word_idx, target_slot=slot, skip_references=True):
                        break
                    continue

            word_idx = self.manifest.find_next_word_index(0)
            if word_idx is None:
                self._update_engine(
                    state="complete",
                    phase=Phase.COMPLETE.value,
                    message="All clips collected",
                )
                break

            word = self._vocab[word_idx]["word"]
            if not self._collect_word(word, word_idx):
                break

        self._update_engine(state="idle", phase=Phase.IDLE.value)

    def _collect_word(
        self,
        word: str,
        word_idx: int,
        target_slot: int | None = None,
        skip_references: bool = False,
    ) -> bool:
        refs_before_next_record = skip_references
        pending_slot: int | None = target_slot

        while not self._stop.is_set():
            if not self._wait_if_paused():
                return False

            if self._replay_references_on_resume:
                self._replay_references_on_resume = False
                refs_before_next_record = False

            if not refs_before_next_record:
                if not self._play_references(word, word_idx):
                    return False
                refs_before_next_record = True

            if pending_slot is not None:
                slot = pending_slot
                pending_slot = None
            else:
                slot = self.manifest.next_incomplete_slot(word)
                if slot is None:
                    return True

            result = self._record_slot(word, word_idx, slot)
            if result == "replay_same":
                refs_before_next_record = False
                pending_slot = slot
                continue
            if result == "replay_next":
                refs_before_next_record = False
                continue
            if not result:
                return False

            if self.manifest.word_is_complete(word):
                return True

        return False

    def _record_slot(self, word: str, word_idx: int, slot: int) -> bool | str:
        """Record a fixed-length clip immediately — no motion trigger."""
        target_frames = max(1, round(RECORD_DURATION_SEC * self.webcam.fps))
        frame_interval = 1.0 / max(self.webcam.fps, 1.0)

        self._update_engine(
            current_word=word,
            current_word_index=word_idx,
            current_slot=slot,
            phase=Phase.RECORDING.value,
            phase_duration_sec=RECORD_DURATION_SEC,
            motion=0.0,
            message=(
                f"Recording {RECORD_DURATION_SEC:.1f}s — sample {slot + 1}/{SAMPLES_PER_WORD}. "
                f"Sign now!"
            ),
        )

        frames: list[np.ndarray] = []
        next_frame_time = time.monotonic()

        while len(frames) < target_frames and not self._stop.is_set():
            if not self._wait_if_paused():
                return False
            if self._replay_references_on_resume:
                return "replay_same"

            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(min(0.05, next_frame_time - now))
                continue
            next_frame_time += frame_interval

            frame = self.webcam.read()
            if frame is not None:
                frames.append(frame.copy())

        if len(frames) < target_frames:
            if self._replay_references_on_resume:
                return "replay_same"
            return False

        word_dir = COLLECTION_OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{word}_{slot:04d}.mp4"
        out_path = word_dir / fname

        writer = self.webcam.writer_fourcc_path(str(out_path))
        for f in frames[:target_frames]:
            writer.write(f)
        writer.release()

        reencode_for_browser(out_path)

        duration_ms = int(RECORD_DURATION_SEC * 1000)
        self.manifest.mark_slot_complete(word, slot, fname, duration_ms)

        cooldown = self.manifest.get_word_timing(word)["cooldown_sec"]
        cooldown_label = f"{cooldown:g}s"
        self._update_engine(
            phase=Phase.COOLDOWN.value,
            phase_duration_sec=cooldown,
            motion=0.0,
            message=(
                f"Saved {fname} ({RECORD_DURATION_SEC:.1f}s) — "
                + (
                    f"next clip in {cooldown_label}"
                    if cooldown > 0
                    else "next clip starting now"
                )
            ),
            total_completed=self.manifest.snapshot()["engine"]["total_completed"],
        )
        if cooldown <= 0:
            return True
        if not self._sleep_interruptible(cooldown):
            if self._replay_references_on_resume:
                return "replay_next"
            return False
        return True
