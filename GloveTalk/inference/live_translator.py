import json
import sys
import time
from pathlib import Path

import numpy as np
import pyttsx3
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.feature_utils import GloveFeatureBuffer, load_scaler
from inference.glove_io import DEFAULT_GLOVE_TCP_PORT, TCPSerial, parse_glove_line
from paths import FEATURE_CONFIG, WORDS_CLASSES, WORDS_MODEL, WORDS_SCALER

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
TCP_PORT = DEFAULT_GLOVE_TCP_PORT

CALIBRATION_FILE = ROOT / "sensor_calibration.json"

CONFIDENCE_THRESHOLD = 0.75
REQUIRED_CONSECUTIVE_MATCHES = 3
BUFFER_CLEAR_INTERVAL_SEC = 3.0

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 1.0)


def speak(text):
    print(f"\nAI UTTERANCE: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


with open(CALIBRATION_FILE) as f:
    cal_data = json.load(f)
    baseline_min = cal_data["baseline_min"]
    baseline_max = cal_data["baseline_max"]

print("Loading words Bi-LSTM model...")
model = tf.keras.models.load_model(WORDS_MODEL)
classes = np.load(WORDS_CLASSES, allow_pickle=True)
scaler = load_scaler(WORDS_SCALER)
feature_buffer = GloveFeatureBuffer(window_size=WINDOW_SIZE)

print(f"Starting Wi-Fi receiver on port {TCP_PORT}...")
try:
    ser = TCPSerial(port=TCP_PORT)
    time.sleep(2)
except Exception as exc:
    print(f"Connection error: {exc}")
    raise SystemExit(1)

ser.reset_input_buffer()
print(f"Listening on port {TCP_PORT} (locked for this session)")
print(f"Classes ({len(classes)}): {', '.join(classes)}")
print("=== SYSTEM ONLINE: SHOW GESTURES NATURALLY ===")

consecutive_predictions = []
last_status = time.time()
last_buffer_clear = time.time()

print("Waiting for glove data...", flush=True)
stream_deadline = time.time() + 10.0
while time.time() < stream_deadline:
    if ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore")
        if parse_glove_line(line):
            print("Glove stream OK — start signing.\n", flush=True)
            ser.reset_input_buffer()
            break
    time.sleep(0.01)
else:
    print("ERROR: No glove data received. Check both gloves are on and Wi-Fi is connected.")
    ser.close()
    raise SystemExit(1)


def normalize_frame(raw_data):
    normalized = []
    for i in range(len(raw_data)):
        if i in IMU_INDICES:
            normalized.append(raw_data[i])
        else:
            val_min, val_max = baseline_min[i], baseline_max[i]
            scaled = 0.0 if val_max == val_min else (raw_data[i] - val_min) / (val_max - val_min)
            normalized.append(round(max(0.0, min(1.0, scaled)), 4))
    return normalized


while True:
    try:
        if not ser.in_waiting:
            time.sleep(0.001)
            continue

        line = ser.readline().decode("utf-8", errors="ignore")
        parsed = parse_glove_line(line)
        if not parsed:
            continue

        now = time.time()
        if now - last_buffer_clear >= BUFFER_CLEAR_INTERVAL_SEC:
            feature_buffer.clear()
            consecutive_predictions.clear()
            last_buffer_clear = now

        feature_buffer.add_frame(normalize_frame(parsed))

        if not feature_buffer.is_ready():
            if time.time() - last_status > 1.0:
                print(f"Buffering... {len(feature_buffer._raw_buffer)}/{WINDOW_SIZE} frames", flush=True)
                last_status = time.time()
            continue

        input_matrix = feature_buffer.get_feature_window(scaler)
        predictions = model.predict(input_matrix, verbose=0)[0]
        max_prob = float(np.max(predictions))
        predicted_label = classes[np.argmax(predictions)]
        print(f"Live: {predicted_label} ({max_prob * 100:.1f}%)", flush=True)

        if max_prob > CONFIDENCE_THRESHOLD:
            consecutive_predictions.append(predicted_label)
        else:
            consecutive_predictions.clear()

        if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_MATCHES:
            if len(set(consecutive_predictions)) == 1:
                speak(predicted_label)
                consecutive_predictions.clear()
                feature_buffer.clear()
                last_buffer_clear = time.time()
                time.sleep(1.0)
                ser.reset_input_buffer()
            else:
                consecutive_predictions.pop(0)
    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print(f"\nGlitch bypassed: {e}", flush=True)
        ser.reset_input_buffer()

ser.close()
