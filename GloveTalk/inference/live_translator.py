import json
import sys
import time
from pathlib import Path

import numpy as np
import pyttsx3
import serial
import serial.tools.list_ports
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.feature_utils import GloveFeatureBuffer, load_scaler
from paths import FEATURE_CONFIG, WORDS_CLASSES, WORDS_MODEL, WORDS_SCALER

WINDOW_SIZE = 30
NUM_RAW_FEATURES = 18
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}


def find_esp_port():
    print("Scanning for hardware...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "CP210" in port.description or "CH340" in port.description or "UART" in port.description:
            print(f"Found device on {port.device}")
            return port.device
    if ports:
        return ports[0].device
    return None


SERIAL_PORT = find_esp_port()
if not SERIAL_PORT:
    print("No serial port found.")
    raise SystemExit(1)

BAUD_RATE = 115200
CALIBRATION_FILE = ROOT / "sensor_calibration.json"

CONFIDENCE_THRESHOLD = 0.90
REQUIRED_CONSECUTIVE_MATCHES = 3

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

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
print(f"Connected on {SERIAL_PORT}")
print("=== SYSTEM ONLINE: SHOW GESTURES NATURALLY ===")
ser.reset_input_buffer()

consecutive_predictions = []
loop_counter = 0


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
                loop_counter += 1

                if feature_buffer.is_ready() and loop_counter % 8 == 0:
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
                            speak(predicted_label)
                            consecutive_predictions.clear()
                            feature_buffer.clear()
                            time.sleep(1.5)
                            ser.reset_input_buffer()
                        else:
                            consecutive_predictions.pop(0)
    except Exception as e:
        print(f"\nGlitch bypassed: {e}")
        ser.reset_input_buffer()
