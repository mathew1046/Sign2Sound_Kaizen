"""Shared webcam capture and MJPEG streaming."""

from __future__ import annotations

import threading
import time
from typing import Iterator

import cv2
import numpy as np

from dashboard.config import CAM_FPS, CAM_HEIGHT, CAM_WIDTH, CAMERA_ENABLED, CAMERA_INDEX


class WebcamCapture:
    """Thread-safe single-camera owner with latest-frame buffer."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAM_WIDTH,
        height: int = CAM_HEIGHT,
        fps: float = CAM_FPS,
        mirror: bool = True,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.enabled = CAMERA_ENABLED
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._latest: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.ok = False
        self.error: str | None = None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            if enabled == self.enabled:
                return
            self.enabled = enabled
        if not enabled:
            self.stop()
        else:
            with self._lock:
                self.error = None

    def open(self) -> bool:
        if not self.enabled:
            self.error = "Camera disabled"
            self.ok = False
            return False
        with self._lock:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                self.error = f"Cannot open camera {self.camera_index}"
                self.ok = False
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = actual_fps if actual_fps and actual_fps > 0 else self.fps
            self.ok = True
            self.error = None
            return True

    def start_reader(self) -> None:
        if self._running:
            return
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="webcam")
        self._thread.start()

    def _read_loop(self) -> None:
        interval = 1.0 / max(self.fps, 1.0)
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                cap = self._cap
            if cap is None or not cap.isOpened():
                break
            ok, frame = cap.read()
            if ok:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                ok_enc, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                jpeg = buf.tobytes() if ok_enc else None
                with self._lock:
                    self._latest = frame
                    if jpeg:
                        self._latest_jpeg = jpeg
            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def ensure_reader(self) -> bool:
        """Open camera and start capture thread if needed."""
        if not self.enabled:
            self.error = "Camera disabled"
            self.ok = False
            return False
        if not self.ok:
            if not self.open():
                return False
        self.start_reader()
        return self.ok

    def wait_for_jpeg(self, timeout: float = 3.0) -> bytes | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jpeg = self.get_jpeg()
            if jpeg:
                return jpeg
            time.sleep(0.05)
        return None

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def mjpeg_stream(self) -> Iterator[bytes]:
        boundary = b"frame"
        while self._running or self.get_jpeg():
            jpeg = self.get_jpeg()
            if jpeg:
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(1.0 / max(self.fps, 1.0))

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        self.ok = False

    def writer_fourcc_path(self, path: str) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
