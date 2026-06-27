"""Live GloveTalk inference over TCP.

Input:
    Hardware connects to TCP port 8080 and sends one sensor frame per line.

Output:
    Clients connect to TCP port 8081 and receive newline-delimited JSON
    prediction events containing the raw model probability vector.
"""
import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.feature_utils import GloveFeatureBuffer, load_scaler
from inference.glove_io import DEFAULT_GLOVE_TCP_PORT, TCPLineReceiver, parse_glove_line
from paths import FEATURE_CONFIG, WORDS_CLASSES, WORDS_MODEL, WORDS_SCALER

DEFAULT_INPUT_PORT = DEFAULT_GLOVE_TCP_PORT
DEFAULT_FEED_PORT = 8081
DEFAULT_HOST = "0.0.0.0"

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
CALIBRATION_FILE = ROOT / "sensor_calibration.json"

CONFIDENCE_THRESHOLD = 0.75
REQUIRED_CONSECUTIVE_MATCHES = 3


class PredictionFeedServer:
    """Broadcast newline-delimited JSON prediction events to TCP clients."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()

    def broadcast(self, event: dict) -> None:
        payload = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        with self._clients_lock:
            live_clients = []
            for client in self._clients:
                try:
                    client.sendall(payload)
                    live_clients.append(client)
                except OSError:
                    try:
                        client.close()
                    except OSError:
                        pass
            self._clients = live_clients

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            server.settimeout(0.5)
            print(f"Prediction feed listening on {self.host}:{self.port}")

            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._clients_lock:
                    self._clients.append(conn)
                print(f"Feed client connected from {addr[0]}:{addr[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live GloveTalk TCP inference.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address for both TCP servers.")
    parser.add_argument("--input-port", type=int, default=DEFAULT_INPUT_PORT, help="Hardware input TCP port.")
    parser.add_argument("--feed-port", type=int, default=DEFAULT_FEED_PORT, help="Prediction output TCP port.")
    parser.add_argument("--predict-every", type=int, default=1, help="Run inference every N valid frames.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top labels included in each event.")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Stable label confidence threshold.")
    return parser.parse_args()


def normalize_frame(raw_data: list[float], baseline_min: list[float], baseline_max: list[float]) -> list[float]:
    normalized = []
    for index, value in enumerate(raw_data):
        if index in IMU_INDICES:
            normalized.append(round(value, 6))
            continue

        val_min = baseline_min[index]
        val_max = baseline_max[index]
        scaled = 0.0 if val_max == val_min else (value - val_min) / (val_max - val_min)
        normalized.append(round(max(0.0, min(1.0, scaled)), 4))
    return normalized


def load_window_size() -> int:
    if not FEATURE_CONFIG.exists():
        return WINDOW_SIZE
    with open(FEATURE_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    return int(config.get("window_size", WINDOW_SIZE))


def get_lan_hint(port: int) -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return f"{probe.getsockname()[0]}:{port}"
    except OSError:
        return f"<this-computer-ip>:{port}"


def build_prediction_event(
    sequence: int,
    classes: np.ndarray,
    predictions: np.ndarray,
    top_k: int,
    stable_label: str | None,
) -> dict:
    scores = [float(score) for score in predictions]
    max_index = int(np.argmax(predictions))
    ranked_indices = np.argsort(predictions)[::-1][:top_k]
    labels = [str(label) for label in classes.tolist()]

    return {
        "type": "prediction",
        "timestamp": time.time(),
        "sequence": sequence,
        "label": labels[max_index],
        "confidence": scores[max_index],
        "stable_label": stable_label,
        "classes": labels,
        "scores": scores,
        "probabilities": dict(zip(labels, scores)),
        "top_k": [
            {"label": labels[int(index)], "score": scores[int(index)]}
            for index in ranked_indices
        ],
    }


def main() -> None:
    args = parse_args()

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
        baseline_min = cal_data["baseline_min"]
        baseline_max = cal_data["baseline_max"]

    window_size = load_window_size()
    print(f"Loading latest words hardware model: {WORDS_MODEL}")
    model = tf.keras.models.load_model(WORDS_MODEL)
    classes = np.load(WORDS_CLASSES, allow_pickle=True)
    scaler = load_scaler(WORDS_SCALER)
    feature_buffer = GloveFeatureBuffer(window_size=window_size)

    receiver = TCPLineReceiver(args.host, args.input_port)
    feed = PredictionFeedServer(args.host, args.feed_port)
    receiver.start()
    feed.start()

    print("=== SYSTEM ONLINE: SEND GLOVE FRAMES TO TCP PORT 8080 ===")
    print(f"From another laptop, read predictions at {get_lan_hint(args.feed_port)}")
    print("Each feed line is JSON with full model scores.")

    consecutive_predictions: list[str] = []
    valid_frames = 0
    prediction_sequence = 0

    try:
        while True:
            line = receiver.readline(timeout=0.1)
            if line is None:
                continue

            parsed = parse_glove_line(line)
            if parsed is None or len(parsed) != NUM_RAW_FEATURES:
                continue

            feature_buffer.add_frame(normalize_frame(parsed, baseline_min, baseline_max))
            valid_frames += 1

            if not feature_buffer.is_ready() or valid_frames % max(args.predict_every, 1) != 0:
                continue

            input_matrix = feature_buffer.get_feature_window(scaler)
            predictions = model.predict(input_matrix, verbose=0)[0]
            max_prob = float(np.max(predictions))
            predicted_label = str(classes[int(np.argmax(predictions))])
            stable_label = None

            if max_prob > args.threshold:
                consecutive_predictions.append(predicted_label)
            else:
                consecutive_predictions.clear()

            if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_MATCHES:
                if len(set(consecutive_predictions)) == 1:
                    stable_label = predicted_label
                    consecutive_predictions.clear()
                else:
                    consecutive_predictions.pop(0)

            prediction_sequence += 1
            event = build_prediction_event(
                prediction_sequence,
                classes,
                predictions,
                max(args.top_k, 1),
                stable_label,
            )
            feed.broadcast(event)
            print(f"Live Prediction: {predicted_label} ({max_prob * 100:.1f}%)", end="\r")
    except KeyboardInterrupt:
        print("\nStopping live translator...")
    finally:
        receiver.stop()
        feed.stop()


if __name__ == "__main__":
    main()
