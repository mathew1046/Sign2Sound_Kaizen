import json
import sys
import time
from pathlib import Path

import numpy as np
import pyttsx3
import serial
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.feature_utils import GloveFeatureBuffer, load_scaler
from paths import FEATURE_CONFIG, WORDS_CLASSES, WORDS_MODEL, WORDS_SCALER

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
BAUD_RATE = 115200
GLOVE_SERIAL_PORT = "/dev/ttyUSB0"


def parse_line(line):
    try:
        line = line.strip()
        if not line or "|" not in line:
            return None
        parts = line.split("|")
        if len(parts) < 2:
            return None
        l_data = parts[0].strip().split(",")[1:]
        r_data = parts[1].strip().split(",")[1:]
        values = [float(x) for x in l_data] + [float(x) for x in r_data]
        if len(values) != NUM_RAW_FEATURES:
            return None
        return values
    except Exception:
        return None


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

print(f"Opening glove port {GLOVE_SERIAL_PORT}...")
try:
    ser = serial.Serial(GLOVE_SERIAL_PORT, BAUD_RATE, timeout=0.5)
except (serial.SerialException, OSError) as exc:
    print(f"Cannot open {GLOVE_SERIAL_PORT}: {exc}")
    raise SystemExit(1)

time.sleep(0.3)
ser.reset_input_buffer()
print(f"Connected on {GLOVE_SERIAL_PORT} (locked for this session)")
print(f"Classes ({len(classes)}): {', '.join(classes)}")
print("=== SYSTEM ONLINE: SHOW GESTURES NATURALLY ===")

consecutive_predictions = []
last_status = time.time()
last_buffer_clear = time.time()

print("Waiting for glove data...", flush=True)
stream_deadline = time.time() + 10.0
while time.time() < stream_deadline:
    line = ser.readline().decode("utf-8", errors="ignore")
    if parse_line(line):
        print("Glove stream OK — start signing.\n", flush=True)
        ser.reset_input_buffer()
        break
else:
    print("ERROR: No glove data received. Check both gloves are on and USB is connected.")
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
        line = ser.readline().decode("utf-8", errors="ignore")
        if not line:
            time.sleep(0.001)
            continue

        parsed = parse_line(line)
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
