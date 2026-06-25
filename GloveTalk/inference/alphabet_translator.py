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
from paths import ALPHABET_CLASSES, ALPHABET_MODEL, ALPHABET_SCALER

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}
BAUD_RATE = 115200
GLOVE_SERIAL_PORT = "/dev/ttyUSB0"
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

print(f"Opening glove port {GLOVE_SERIAL_PORT}...")
try:
    ser = serial.Serial(GLOVE_SERIAL_PORT, BAUD_RATE, timeout=0.1)
except (serial.SerialException, OSError) as exc:
    print(f"Cannot open {GLOVE_SERIAL_PORT}: {exc}")
    raise SystemExit(1)

print(f"Connected on {GLOVE_SERIAL_PORT} (locked for this session)")
print("=== SYSTEM ONLINE: SPELL ALPHABETS NATURALLY ===")
ser.reset_input_buffer()

consecutive_predictions = []
last_spoken_label = None
last_spoken_time = 0


def parse_line(line):
    try:
        parts = line.split("|")
        l_data = parts[0].split(",")[1:]
        r_data = parts[1].split(",")[1:]
        return [float(x) for x in l_data] + [float(x) for x in r_data]
    except Exception:
        return None


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
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            parsed = parse_line(line)
            if parsed and len(parsed) == NUM_RAW_FEATURES:
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
    except Exception as e:
        print(f"\nGlitch bypassed: {e}")
        ser.reset_input_buffer()
