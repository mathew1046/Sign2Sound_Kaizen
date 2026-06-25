import serial
import serial.tools.list_ports
import time
import json
import numpy as np
import pyttsx3
import joblib
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

def find_esp_port():
    print("🔍 Scanning for hardware...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "CP210" in port.description or "CH340" in port.description or "UART" in port.description:
            print(f"✅ Found recognized device on {port.device}")
            return port.device
    if ports:
        print(f"⚠️ Specific chip not found. Defaulting to: {ports[0].device}")
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
MODEL_FILENAME = "alphabet_model.pkl"
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
try:
    with open(CALIBRATION_FILE, 'r') as f:
        cal_data = json.load(f)
        baseline_min = cal_data['baseline_min']
        baseline_max = cal_data['baseline_max']
except FileNotFoundError:
    print(f"❌ Error: {CALIBRATION_FILE} missing! Run your data collector first.")
    exit()

# --- LOAD AI NEURAL NETWORK ---
print("🧠 Loading Alphabet Random Forest Model...")
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ Model online! Vocabulary: {model.classes_}")
except FileNotFoundError:
    print(f"❌ Error: {MODEL_FILENAME} not found! Run train_alphabets.py first.")
    exit()

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
print("\n=== SYSTEM ONLINE: SPELL ALPHABETS NATURALLY ===")
ser.reset_input_buffer()

# --- TUNING PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.85      # 85% certainty required to speak
REQUIRED_CONSECUTIVE_MATCHES = 4 # Must predict the same letter 4 times in a row to speak
COOLDOWN_TIME = 1.5              # Seconds to wait before allowing the SAME letter to be spoken again

consecutive_predictions = []
last_spoken_label = None
last_spoken_time = 0

while True:
    try:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parsed = parse_line(line)
            
            if parsed and len(parsed) == NUM_FEATURES:
                normalized = normalize_frame(parsed)
                
                # Random Forest expects a 2D array: [ [features] ]
                input_matrix = np.array([normalized])
                
                # Run the AI prediction
                probs = model.predict_proba(input_matrix)[0]
                max_prob = np.max(probs)
                predicted_label = model.classes_[np.argmax(probs)]
                
                # Debug print
                print(f"Live Prediction: {predicted_label} ({max_prob*100:.1f}%)", end='\r')
                
                if max_prob > CONFIDENCE_THRESHOLD:
                    consecutive_predictions.append(predicted_label)
                else:
                    consecutive_predictions.clear()
                
                # If the prediction remains stable across multiple frames
                if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_MATCHES:
                    
                    # Ensure all consecutive frames are exactly the same letter
                    if len(set(consecutive_predictions)) == 1:
                        
                        current_time = time.time()
                        
                        # Logic to prevent spamming: 
                        # Speak if it's a NEW letter, OR if the cooldown time has passed for the SAME letter.
                        if predicted_label != last_spoken_label or (current_time - last_spoken_time) > COOLDOWN_TIME:
                            
                            speak(predicted_label)
                            
                            last_spoken_label = predicted_label
                            last_spoken_time = current_time
                            
                            # HOUSEKEEPING: Clear states and flush buffer
                            consecutive_predictions.clear()
                            ser.reset_input_buffer() 
                            
                    else:
                        consecutive_predictions.pop(0)
                        
    except Exception as e:
        print(f"\n⚠️ Math glitch bypassed: {e}")
        ser.reset_input_buffer()
        continue