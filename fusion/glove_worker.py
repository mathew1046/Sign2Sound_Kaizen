"""Background glove TCP reader with anti-FP gates."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from fusion.tokens import SignToken

GLOVE_ROOT = Path(__file__).resolve().parents[1] / "GloveTalk"

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
FLEX_INDICES = [i for i in range(NUM_RAW_FEATURES) if i not in IMU_INDICES]
BUFFER_CLEAR_INTERVAL_SEC = 3.0
DEFAULT_GLOVE_TCP_PORT = 8080


class GloveWorker:
    """Read glove TCP stream and emit SignToken objects on a queue."""

    def __init__(
        self,
        tcp_port: int = DEFAULT_GLOVE_TCP_PORT,
        *,
        margin_threshold: float = 0.25,
        activity_threshold: float = 0.02,
        consecutive_required: int = 5,
        calibration_file: Path | None = None,
        connect_timeout_sec: float = 15.0,
    ):
        self.tcp_port = tcp_port
        self.margin_threshold = margin_threshold
        self.activity_threshold = activity_threshold
        self.consecutive_required = consecutive_required
        self.calibration_file = calibration_file or (GLOVE_ROOT / "sensor_calibration.json")
        self.connect_timeout_sec = connect_timeout_sec

        self._queue: queue.Queue[SignToken] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = False
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, daemon=True, name="glove-worker")
        self._thread.start()
        deadline = time.time() + self.connect_timeout_sec
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

    def health(self) -> dict:
        """Return worker health status."""
        return {
            "available": self._available,
            "error": self._error,
            "queue_depth": self._queue.qsize(),
        }

    def _run(self) -> None:
        ser = None
        try:
            glove_root = str(GLOVE_ROOT.resolve())
            if glove_root not in sys.path:
                sys.path.insert(0, glove_root)

            import tensorflow as tf  # noqa: WPS433

            from inference.feature_utils import GloveFeatureBuffer, load_scaler  # noqa: WPS433
            from inference.glove_io import TCPSerial, parse_glove_line  # noqa: WPS433
            from paths import WORDS_CLASSES, WORDS_MODEL, WORDS_SCALER  # noqa: WPS433

            with open(self.calibration_file) as f:
                cal_data = json.load(f)
            baseline_min = cal_data["baseline_min"]
            baseline_max = cal_data["baseline_max"]

            model = tf.keras.models.load_model(WORDS_MODEL)
            classes = np.load(WORDS_CLASSES, allow_pickle=True)
            scaler = load_scaler(WORDS_SCALER)
            feature_buffer = GloveFeatureBuffer(window_size=WINDOW_SIZE)

            # Validate fusion vocabulary against actual glove classes
            try:
                from fusion.vocabulary import FusionVocabulary

                vocab = FusionVocabulary()
                unmapped = vocab.validate_against_glove_classes([str(c) for c in classes])
                if unmapped:
                    print(f"[glove] WARNING: fusion vocabulary missing mappings for: {unmapped}")
                else:
                    print(f"[glove] fusion vocabulary validated: all {len(classes)} classes mapped")
            except Exception as exc:
                print(f"[glove] fusion vocabulary validation skipped: {exc}")

            ser = TCPSerial(port=self.tcp_port)
            time.sleep(0.5)

            stream_deadline = time.time() + self.connect_timeout_sec
            while time.time() < stream_deadline and not self._stop.is_set():
                if ser.in_waiting:
                    line = ser.readline().decode("utf-8", errors="ignore")
                    if parse_glove_line(line):
                        ser.reset_input_buffer()
                        self._available = True
                        print(f"[glove] TCP stream OK on port {self.tcp_port}")
                        break
                time.sleep(0.01)
            else:
                if not self._stop.is_set():
                    raise TimeoutError(
                        f"No glove data on TCP port {self.tcp_port} within {self.connect_timeout_sec}s"
                    )

            consecutive: list[str] = []
            last_buffer_clear = time.time()
            raw_flex_window: list[list[float]] = []

            while not self._stop.is_set():
                if not ser.in_waiting:
                    time.sleep(0.001)
                    continue

                line = ser.readline().decode("utf-8", errors="ignore")
                parsed = parse_glove_line(line)
                if not parsed:
                    continue

                now = time.monotonic()
                if now - last_buffer_clear >= BUFFER_CLEAR_INTERVAL_SEC:
                    feature_buffer.clear()
                    consecutive.clear()
                    raw_flex_window.clear()
                    last_buffer_clear = now

                normalized = []
                for i, val in enumerate(parsed):
                    if i in IMU_INDICES:
                        normalized.append(val)
                    else:
                        val_min, val_max = baseline_min[i], baseline_max[i]
                        scaled = 0.0 if val_max == val_min else (val - val_min) / (val_max - val_min)
                        normalized.append(round(max(0.0, min(1.0, scaled)), 4))

                raw_flex_window.append([normalized[i] for i in FLEX_INDICES])
                if len(raw_flex_window) > WINDOW_SIZE:
                    raw_flex_window.pop(0)

                feature_buffer.add_frame(normalized)
                if not feature_buffer.is_ready():
                    continue

                input_matrix = feature_buffer.get_feature_window(scaler)
                predictions = model.predict(input_matrix, verbose=0)[0]
                top2 = np.partition(predictions, -2)[-2:]
                top2.sort()
                max_prob = float(top2[-1])
                second_prob = float(top2[-2]) if len(top2) > 1 else 0.0
                margin = max_prob - second_prob
                predicted_label = str(classes[np.argmax(predictions)])

                flex_std = 0.0
                if raw_flex_window:
                    flex_arr = np.array(raw_flex_window, dtype=np.float32)
                    flex_std = float(np.std(flex_arr))

                if (
                    margin >= self.margin_threshold
                    and flex_std >= self.activity_threshold
                    and predicted_label != "rest"
                ):
                    consecutive.append(predicted_label)
                else:
                    consecutive.clear()

                if len(consecutive) >= self.consecutive_required:
                    if len(set(consecutive)) == 1:
                        token = SignToken(
                            gloss=predicted_label,
                            source="glove",
                            confidence=max_prob,
                            timestamp=now,
                            meta={
                                "margin": margin,
                                "flex_std": flex_std,
                                "consecutive": len(consecutive),
                            },
                        )
                        try:
                            self._queue.put_nowait(token)
                        except queue.Full:
                            pass
                        consecutive.clear()
                        feature_buffer.clear()
                        raw_flex_window.clear()
                        last_buffer_clear = now
                        ser.reset_input_buffer()
                    else:
                        consecutive.pop(0)

            if ser is not None:
                ser.close()
        except Exception as exc:
            self._error = str(exc)
            self._available = False
            print(f"[glove] disabled: {exc}")
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
