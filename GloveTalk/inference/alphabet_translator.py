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
from paths import ALPHABET_CLASSES, ALPHABET_MODEL, ALPHABET_SCALER

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
TCP_PORT = DEFAULT_GLOVE_TCP_PORT
CALIBRATION_FILE = ROOT / "sensor_calibration.json"

CONFIDENCE_THRESHOLD = 0.85
REQUIRED_CONSECUTIVE_MATCHES = 4
COOLDOWN_TIME = 1.5

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)


def speak(text):
    print(f"\nAI UTTERANCE: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


with open(CALIBRATION_FILE) as f:
    cal_data = json.load(f)
    baseline_min = cal_data["baseline_min"]
    baseline_max = cal_data["baseline_max"]

model = tf.keras.models.load_model(ALPHABET_MODEL)
classes = np.load(ALPHABET_CLASSES, allow_pickle=True)
scaler = load_scaler(ALPHABET_SCALER)
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
print("=== SYSTEM ONLINE: SPELL ALPHABETS NATURALLY ===")

consecutive_predictions = []
last_spoken_label = None
last_spoken_time = 0

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

        feature_buffer.add_frame(normalize_frame(parsed))

        if feature_buffer.is_ready():
            input_matrix = feature_buffer.get_feature_window(scaler)
            predictions = model.predict(input_matrix, verbose=0)[0]
            max_prob = float(np.max(predictions))
            predicted_label = classes[np.argmax(predictions)]
            print(f"Live Prediction: {predicted_label} ({max_prob * 100:.1f}%)", end="\r")

            if max_prob > CONFIDENCE_THRESHOLD:
                consecutive_predictions.append(predicted_label)
            else:
                consecutive_predictions.clear()

            if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_MATCHES:
                if len(set(consecutive_predictions)) == 1:
                    current_time = time.time()
                    if predicted_label != last_spoken_label or (current_time - last_spoken_time) > COOLDOWN_TIME:
                        speak(predicted_label)
                        last_spoken_label = predicted_label
                        last_spoken_time = current_time
                        consecutive_predictions.clear()
                        ser.reset_input_buffer()
                else:
                    consecutive_predictions.pop(0)
    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print(f"\nGlitch bypassed: {e}")
        ser.reset_input_buffer()

ser.close()
