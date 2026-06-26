"""Background alphabet transformer on shared camera frames."""

from __future__ import annotations

import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fusion.tokens import SignToken

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_ROOT = REPO_ROOT / "alphabet_transformer"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ALPHA_ROOT))

from alphabet_transformer.hand_crop import (  # noqa: E402
    default_cansik_paths,
    extract_frame_landmarks_detailed,
    try_create_cansik_detector,
    crop_frame,
    remap_landmarks_to_full_frame,
    _landmarks_from_results,
    FrameLandmarks,
)
from alphabet_transformer.model import SignTransformer  # noqa: E402
from alphabet_transformer.paths import DEFAULT_WEIGHTS, HAND_LANDMARKER  # noqa: E402


def _get_hand_landmarker():
    if not HAND_LANDMARKER.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, HAND_LANDMARKER)
    base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class AlphabetWorker:
    """Consume latest frames and emit alphabet letter tokens when enabled."""

    def __init__(
        self,
        *,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.85,
        use_crop: bool = True,
        crop_pad: float = 0.15,
        hand_det_confidence: float = 0.2,
        min_interval_sec: float = 0.16,
        device: str | None = None,
    ):
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self.confidence_threshold = confidence_threshold
        self.use_crop = use_crop
        self.crop_pad = crop_pad
        self.hand_det_confidence = hand_det_confidence
        self.min_interval_sec = min_interval_sec
        self.device_preference = device

        self._queue: queue.Queue[SignToken] = queue.Queue(maxsize=32)
        self._frame_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._enabled = False
        self._crop_bbox: tuple[int, int, int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = False
        self._error: str | None = None
        self._external_bbox = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    def set_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self._enabled = enabled
            if not enabled:
                self._crop_bbox = None
                self._external_bbox = None

    def get_crop_bbox(self) -> tuple[int, int, int, int] | None:
        with self._state_lock:
            return self._crop_bbox

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def submit_hand_bbox(self, bbox: tuple[int, int, int, int] | None) -> None:
        with self._state_lock:
            self._external_bbox = bbox

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, daemon=True, name="alphabet-worker")
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._available:
                return True
            if self._error:
                return False
            time.sleep(0.05)
        return self._available

    def poll(self) -> list[SignToken]:
        tokens: list[SignToken] = []
        while True:
            try:
                tokens.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return tokens

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            if self.device_preference:
                device = torch.device(self.device_preference)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            model = SignTransformer(input_dim=126, num_classes=len(classes)).to(device)
            model.load_state_dict(
                torch.load(self.weights_path, map_location=device, weights_only=True)
            )
            model.eval()

            landmarker = _get_hand_landmarker()
            cfg, weights = default_cansik_paths()
            detector = (
                try_create_cansik_detector(cfg, weights, confidence=self.hand_det_confidence)
                if self.use_crop
                else None
            )
            self._available = True
            print(f"[alphabet] worker ready (device={device})")

            frame_queue: deque[np.ndarray] = deque(maxlen=30)
            last_pred_time = 0.0

            while not self._stop.is_set():
                with self._state_lock:
                    enabled = self._enabled

                frame = None
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if not enabled:
                    time.sleep(0.01)
                    continue

                # Check for externally provided bbox first (e.g. from RTMLib)
                with self._state_lock:
                    ext_bbox = self._external_bbox

                if ext_bbox is not None:
                    frame_h, frame_w = frame.shape[:2]
                    x0, y0, w, h = ext_bbox
                    # Clamp bounding box boundaries
                    x0 = max(0, min(x0, frame_w - 1))
                    y0 = max(0, min(y0, frame_h - 1))
                    w = max(1, min(w, frame_w - x0))
                    h = max(1, min(h, frame_h - y0))
                    crop_rect = (x0, y0, w, h)

                    crop_bgr, crop_rect = crop_frame(frame, crop_rect)
                    if crop_bgr.size > 0:
                        input_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        crop_rect = None

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_rgb)
                    results = landmarker.detect(mp_image)
                    landmarks = _landmarks_from_results(results)
                    if crop_rect is not None:
                        landmarks = remap_landmarks_to_full_frame(landmarks, crop_rect, (frame_w, frame_h))
                    result = FrameLandmarks(landmarks=landmarks, crop_bbox=crop_rect)
                else:
                    result = extract_frame_landmarks_detailed(
                        landmarker,
                        frame,
                        use_crop=self.use_crop,
                        detector=detector,
                        pad_frac=self.crop_pad,
                    )

                with self._state_lock:
                    self._crop_bbox = result.crop_bbox

                frame_queue.append(result.landmarks)

                now = time.time()
                if len(frame_queue) == 30 and (now - last_pred_time) >= self.min_interval_sec:
                    data = np.array(frame_queue).reshape(30, 126)
                    mean = data.mean()
                    std = data.std()
                    if std > 1e-6:
                        data = (data - mean) / std

                    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(x)
                        probs = torch.softmax(outputs, dim=1)
                        max_prob, pred_idx = torch.max(probs, 1)
                        predicted = classes[pred_idx.item()]
                        confidence = float(max_prob.item())

                    if confidence >= self.confidence_threshold:
                        token = SignToken(
                            gloss=predicted,
                            source="alphabet",
                            confidence=confidence,
                            timestamp=now,
                            meta={"crop_bbox": result.crop_bbox},
                        )
                        try:
                            self._queue.put_nowait(token)
                        except queue.Full:
                            pass
                        frame_queue.clear()

                    last_pred_time = now

                time.sleep(0.005)
        except Exception as exc:
            self._error = str(exc)
            self._available = False
            print(f"[alphabet] disabled: {exc}")
