import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import serial
import serial.tools.list_ports

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
BAUD_RATE = 115200

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.vocabulary import load_words_vocabulary

CSV_FILENAME = str(ROOT / "data" / "raw" / "sign_language_dataset.csv")
CALIBRATION_FILE = str(ROOT / "sensor_calibration.json")
TARGET_FRAMES = 50

if not SERIAL_PORT:
    print("Exiting...")
    exit()

ABSOLUTE_CSV_PATH = os.path.abspath(CSV_FILENAME)

HEADERS = [
    "timestamp", "label",
    "L_qw", "L_qx", "L_qy", "L_qz", "L_f1", "L_f2", "L_f3", "L_f4", "L_f5",
    "R_qw", "R_qx", "R_qy", "R_qz", "R_f1", "R_f2", "R_f3", "R_f4", "R_f5"
]

print(f"\n📁 FILE SAVE LOCATION: {ABSOLUTE_CSV_PATH}")

# --- FIX 1: Correct header write ---
if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0:
    with open(CSV_FILENAME, mode='w', newline='') as f:
        csv.writer(f).writerow(HEADERS)
    print("📝 Created new dataset file with headers.")
else:
    print("📂 Appending to existing dataset file.")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"🔌 Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting: {e}")
    exit()

def parse_line(line):
    try:
        parts = line.split('|')
        if len(parts) < 2:
            return None
        l_data = parts[0].strip().split(',')[1:]   # skip 'L' label
        r_data = parts[1].strip().split(',')[1:]   # skip 'R' label
        values = [float(x) for x in l_data] + [float(x) for x in r_data]
        if len(values) != 18:                      # sanity check
            return None
        return values
    except:
        return None

# --- CALIBRATION ---
baseline_min = []
baseline_max = []

if os.path.exists(CALIBRATION_FILE):
    print("✅ Saved calibration profile found. Loading automatically...")
    with open(CALIBRATION_FILE, 'r') as f:
        cal_data = json.load(f)
        baseline_min = cal_data['baseline_min']
        baseline_max = cal_data['baseline_max']
else:
    print("\n--- First-Time Hardware Calibration ---")
    input("1. Hold both hands completely FLAT and open. Press ENTER...")
    ser.reset_input_buffer()
    flat_frames = []
    while len(flat_frames) < 30:          # more samples = more stable
        if ser.in_waiting:
            parsed = parse_line(ser.readline().decode('utf-8', errors='ignore').strip())
            if parsed:
                flat_frames.append(parsed)
    baseline_min = [sum(col) / len(col) for col in zip(*flat_frames)]

    input("2. Close both hands into a TIGHT FIST. Press ENTER...")
    ser.reset_input_buffer()
    fist_frames = []
    while len(fist_frames) < 30:
        if ser.in_waiting:
            parsed = parse_line(ser.readline().decode('utf-8', errors='ignore').strip())
            if parsed:
                fist_frames.append(parsed)
    baseline_max = [sum(col) / len(col) for col in zip(*fist_frames)]

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({'baseline_min': baseline_min, 'baseline_max': baseline_max}, f, indent=2)
    print("✅ Calibration saved!")

# --- FIX 3: Correct IMU index check ---
# Data layout: [L_qw, L_qx, L_qy, L_qz, L_f1-f5, R_qw, R_qx, R_qy, R_qz, R_f1-f5]
#  indices:      0     1     2     3     4-8       9     10    11    12    13-17
IMU_INDICES = {0, 1, 2, 3, 9, 10, 11, 12}   # <-- fixed: R_imu starts at 9

def normalize_frame(raw_data):
    normalized = []
    for i, val in enumerate(raw_data):
        if i in IMU_INDICES:
            normalized.append(round(val, 6))
        else:
            val_min = baseline_min[i]
            val_max = baseline_max[i]
            if val_max == val_min:
                scaled = 0.0
            else:
                scaled = (val - val_min) / (val_max - val_min)
            normalized.append(round(max(0.0, min(1.0, scaled)), 4))
    return normalized

# --- FIX 2: Thread-safe frame buffer ---
frame_lock = threading.Lock()
is_recording = False
raw_frames = []
last_hardware_signal = time.time()

def record_serial_background():
    global last_hardware_signal
    while True:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                last_hardware_signal = time.time()
                if is_recording:
                    parsed = parse_line(line)
                    if parsed:
                        with frame_lock:        # lock before appending
                            raw_frames.append(parsed)
        except:
            pass
        time.sleep(0.001)

listener_thread = threading.Thread(target=record_serial_background, daemon=True)
listener_thread.start()

# --- MAIN RECORDING LOOP ---
WORD_VOCABULARY = load_words_vocabulary()
print("\n--- DYNAMIC SIGN DATA COLLECTOR ---")
print(f"Allowed vocabulary: {len(WORD_VOCABULARY)} words")
for i, word in enumerate(WORD_VOCABULARY, 1):
    print(f"  {i:2}. {word}")

while True:
    label = input("\nEnter sign label (must be one of the 22 words) or 'quit': ").strip().lower()
    if label == 'quit':
        break
    if not label:
        continue
    if label not in WORD_VOCABULARY:
        print(f"Invalid label. Choose one of the {len(WORD_VOCABULARY)} words listed above.")
        continue

    take_num = 1

    while True:
        cmd = input(f"\n[Take {take_num}] Press ENTER to START '{label}' (or 'back'): ").strip().lower()
        if cmd == 'back':
            break
        if cmd == 'quit':
            exit()

        if time.time() - last_hardware_signal > 3.0:
            print("\n⚠️ ERROR: Hardware timeout — ESP32 not sending data.")
            break

        # START
        with frame_lock:                    # clear safely
            raw_frames.clear()
        is_recording = True
        print("   🔴 RECORDING... Perform the sign!")

        input("   Press ENTER to STOP...")
        is_recording = False

        # Safe copy immediately after stopping
        with frame_lock:
            captured = list(raw_frames)

        total = len(captured)
        print(f"   Captured {total} frames.")

        if total == 0:
            print("   ⚠️ No data captured. Try again.")
            continue

        # Scale to TARGET_FRAMES
        if total >= TARGET_FRAMES:
            # evenly sample rather than just truncate
            indices = [int(i * total / TARGET_FRAMES) for i in range(TARGET_FRAMES)]
            final_frames = [captured[i] for i in indices]
            print(f"   (Resampled {total} → {TARGET_FRAMES} frames)")
        else:
            final_frames = captured.copy()
            last_frame = final_frames[-1]
            while len(final_frames) < TARGET_FRAMES:
                final_frames.append(last_frame)
            print(f"   (Padded {total} → {TARGET_FRAMES} frames)")

        # Normalize and save with simulated timestamps
        time_step = 0.02 
        rows_written = 0
        try:
            with open(CSV_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                for step, frame in enumerate(final_frames):
                    normalized = normalize_frame(frame)
                    ts = round(step * time_step, 3)
                    writer.writerow([ts, label] + normalized)
                    rows_written += 1
                file.flush()
                os.fsync(file.fileno())
            print(f"   ✅ Take {take_num} saved — {rows_written} rows written to disk.")
            take_num += 1
        except Exception as e:
            print(f"   ❌ Save error: {e}")

print("\nDone. Dataset saved to:", ABSOLUTE_CSV_PATH)
ser.close()