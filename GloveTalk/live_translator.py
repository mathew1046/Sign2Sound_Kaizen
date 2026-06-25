import serial
import serial.tools.list_ports
import time
import json
import os
import numpy as np
import pyttsx3
import tensorflow as tf
from collections import deque

def find_esp_port():
    print("🔍 Scanning for hardware...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Check for common ESP32 USB-to-Serial converter names
        if "CP210" in port.description or "CH340" in port.description or "UART" in port.description:
            print(f"✅ Found recognized device on {port.device} ({port.description})")
            return port.device
            
    # Fallback: if no specific name matches, show available ports or pick the first one
    if ports:
        print(f"⚠️ Specific chip not found. Defaulting to first available: {ports[0].device}")
        return ports[0].device
        
    print("❌ No COM ports found! Is the receiver plugged in?")
    return None

# --- CONFIGURATION ---
SERIAL_PORT = find_esp_port()
if not SERIAL_PORT:
    print("Exiting...")
    exit()

BAUD_RATE = 115200
CALIBRATION_FILE = "sensor_calibration.json"
MODEL_FILENAME = "verb_model.h5"
CLASSES_FILENAME = "verb_classes.npy"

TARGET_FRAMES = 50  
NUM_FEATURES = 18   

# --- INITIALIZE TEXT-TO-SPEECH ---
print("🔊 Initializing Voice Engine...")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  
tts_engine.setProperty('volume', 1.0)

def speak(text):
    print(f"\n📢 AI UTTERANCE: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# --- LOAD CALIBRATION ---
with open(CALIBRATION_FILE, 'r') as f:
    cal_data = json.load(f)
    baseline_min = cal_data['baseline_min']
    baseline_max = cal_data['baseline_max']

# --- LOAD AI NEURAL NETWORK ---
print("🧠 Loading Trained Sign Language Model...")
model = tf.keras.models.load_model(MODEL_FILENAME)
classes = np.load(CLASSES_FILENAME, allow_pickle=True)
print(f"✅ Model online! Vocabulary: {classes}")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"🔌 Connected to hardware on port {SERIAL_PORT}")
except Exception as e:
    print(f"❌ Serial Connection Error: {e}")
    exit()

def parse_line(line):
    try:
        parts = line.split('|')
        l_data = parts[0].split(',')[1:] 
        r_data = parts[1].split(',')[1:] 
        return [float(x) for x in l_data] + [float(x) for x in r_data]
    except:
        return None

def normalize_frame(raw_data):
    normalized = []
    for i in range(len(raw_data)):
        if i in [0, 1, 2, 3, 9, 10, 11, 12]: 
            normalized.append(raw_data[i]) 
        else:
            val_min, val_max = baseline_min[i], baseline_max[i]
            scaled = 0.0 if val_max == val_min else (raw_data[i] - val_min) / (val_max - val_min)
            normalized.append(round(max(0.0, min(1.0, scaled)), 4))
    return normalized

# --- CORE TRANSLATION ENGINE ---
print("\n=== SYSTEM ONLINE: SHOW GESTURES NATURALLY ===")
ser.reset_input_buffer()

sliding_window = deque(maxlen=TARGET_FRAMES)
consecutive_predictions = []

# --- TUNING PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.90      # 90% certainty required to speak
REQUIRED_CONSECUTIVE_MATCHES = 3 # Must predict the same word 3 times in a row to speak
loop_counter = 0

while True:
    try:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parsed = parse_line(line)
            
            if parsed and len(parsed) == NUM_FEATURES:
                normalized = normalize_frame(parsed)
                sliding_window.append(normalized)
                loop_counter += 1
                
                # INFERENCE THROTTLING: Only run the AI when the window is full 
                # AND only once every 8 frames to stop the serial buffer backlog.
                if len(sliding_window) == TARGET_FRAMES and loop_counter % 8 == 0:
                    
                    input_matrix = np.array(sliding_window).reshape(1, TARGET_FRAMES, NUM_FEATURES)
                    
                    # Run the AI prediction
                    predictions = model.predict(input_matrix, verbose=0)[0]
                    max_prob = np.max(predictions)
                    predicted_label = classes[np.argmax(predictions)]
                    
                    # Debug print so you can watch your accuracy scores live in the terminal
                    print(f"Live Prediction: {predicted_label} ({max_prob*100:.1f}%)", end='\r')
                    
                    if max_prob > CONFIDENCE_THRESHOLD:
                        consecutive_predictions.append(predicted_label)
                    else:
                        consecutive_predictions.clear()
                    
                    # If the prediction remains stable across our throttled checks
                    if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_MATCHES:
                        if len(set(consecutive_predictions)) == 1:
                            
                            # 1. Speak the phrase through the laptop speaker
                            speak(predicted_label)
                            
                            # 2. HOUSEKEEPING: Clear tracking states so it doesn't loop double-speak
                            consecutive_predictions.clear()
                            sliding_window.clear()
                            
                            # 3. Give you 1.5 seconds to return your hands to a resting position
                            time.sleep(1.5) 
                            
                            # 4. Wipe out all the unread data that piled up while the laptop was speaking
                            ser.reset_input_buffer() 
                        else:
                            consecutive_predictions.pop(0)
                            
    except Exception as e:
        # Prevents micro-glitches from closing the window automatically
        print(f"\n⚠️ Math glitch bypassed: {e}")
        ser.reset_input_buffer()
        continue