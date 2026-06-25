"""Live inference feature pipeline for GloveTalk Bi-LSTM models."""
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preprocess.feature_engineering import NUM_FEATURES, raw_sequence_to_features


class GloveFeatureBuffer:
    def __init__(self, window_size: int = 30, dt: float = 0.02):
        self.window_size = window_size
        self.dt = dt
        self._raw_buffer = deque(maxlen=window_size)

    def add_frame(self, raw_frame: list | np.ndarray) -> None:
        self._raw_buffer.append(np.array(raw_frame, dtype=np.float32))

    def is_ready(self) -> bool:
        return len(self._raw_buffer) == self.window_size

    def get_feature_window(self, scaler=None) -> np.ndarray | None:
        if not self.is_ready():
            return None
        raw = np.stack(list(self._raw_buffer))
        features = raw_sequence_to_features(raw, dt=self.dt)
        window = features[-self.window_size :]
        if scaler is not None:
            n, f = window.shape
            window = scaler.transform(window.reshape(n, f)).reshape(n, f).astype(np.float32)
        return window.reshape(1, self.window_size, NUM_FEATURES)

    def clear(self) -> None:
        self._raw_buffer.clear()


def load_scaler(scaler_path: Path):
    return joblib.load(scaler_path)
