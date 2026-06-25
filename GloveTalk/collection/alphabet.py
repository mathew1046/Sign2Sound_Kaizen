import serial
import csv
import time
import json
import os

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CSV_FILENAME = str(ROOT / "data" / "raw" / "alphabet_dataset.csv")
CALIBRATION_FILE = str(ROOT / "sensor_calibration.json")
SAMPLES_PER_BATCH = 20  

HEADERS = [
    "label",
    "L_qw", "L_qx", "L_qy", "L_qz", "L_f1", "L_f2", "L_f3", "L_f4", "L_f5",
    "R_qw", "R_qx", "R_qy", "R_qz", "R_f1", "R_f2", "R_f3", "R_f4", "R_f5"
]

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting: {e}")
    exit()

with open(CSV_FILENAME, mode='a', newline='') as file:
    writer = csv.writer(file)
    file.seek(0, 2) 
    if file.tell() == 0: writer.writerow(HEADERS)

def parse_line(line):
    try:
        parts = line.split('|')
        l_data = parts[0].split(',')[1:] 
        r_data = parts[1].split(',')[1:] 
        return [float(x) for x in l_data] + [float(x) for x in r_data]
    except:
        return None

# --- AUTOMATIC CALIBRATION PHASE ---
baseline_min = []
baseline_max = []

if os.path.exists(CALIBRATION_FILE):
    print("\n✅ Saved calibration profile found. Loading automatically...")
    with open(CALIBRATION_FILE, 'r') as f:
        cal_data = json.load(f)
        baseline_min = cal_data['baseline_min']
        baseline_max = cal_data['baseline_max']
else:
    print("\n--- First-Time Hardware Calibration ---")
    input("1. Hold both hands completely FLAT and open. Press ENTER...")
    ser.reset_input_buffer() 
    flat_frames = []
    while len(flat_frames) < 10:
        if ser.in_waiting:
            parsed = parse_line(ser.readline().decode('utf-8', errors='ignore').strip())
            if parsed: flat_frames.append(parsed)
    baseline_min = [sum(col)/len(col) for col in zip(*flat_frames)]

    input("2. Close both hands into a TIGHT FIST. Press ENTER...")
    ser.reset_input_buffer() 
    fist_frames = []
    while len(fist_frames) < 10:
        if ser.in_waiting:
            parsed = parse_line(ser.readline().decode('utf-8', errors='ignore').strip())
            if parsed: fist_frames.append(parsed)
    baseline_max = [sum(col)/len(col) for col in zip(*fist_frames)]
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({'baseline_min': baseline_min, 'baseline_max': baseline_max}, f)
    print("✅ Calibration Saved! You will not need to do this again.")

def normalize_frame(raw_data):
    normalized = []
    for i in range(len(raw_data)):
        if i in [0, 1, 2, 3, 9, 10, 11, 12]: normalized.append(raw_data[i])
        else:
            val_min, val_max = baseline_min[i], baseline_max[i]
            scaled = 0.0 if val_max == val_min else (raw_data[i] - val_min) / (val_max - val_min)
            normalized.append(round(max(0.0, min(1.0, scaled)), 4))
    return normalized

# --- CONTINUOUS ALPHABET BATCH LOOP ---
print("\n--- STATIC ALPHABET COLLECTOR ---")
while True:
    label = input("\nEnter the Letter (e.g., 'A') or 'quit': ").strip().upper()
    if label == 'QUIT': break
    if not label: continue

    input(f"Hold '{label}'. Press ENTER to instantly capture {SAMPLES_PER_BATCH} variations...")
    print("🔴 Capturing... (slightly wiggle your hand for natural variation)")
    
    hardware_failed = False
    
    for i in range(SAMPLES_PER_BATCH):
        if hardware_failed: 
            break
            
        ser.reset_input_buffer()
        frames = []
        timeout_start = time.time() # Start a stopwatch
        
        while len(frames) < 3:
            # TIMEOUT CHECK: If 2 seconds pass with no data, hardware is broken
            if time.time() - timeout_start > 2.0:
                print("\n⚠️ ERROR: Hardware Timeout! The ESP32 stopped sending data.")
                print("Check your wiring, reset the ESP32, and try again.")
                hardware_failed = True
                break
                
            if ser.in_waiting:
                parsed = parse_line(ser.readline().decode('utf-8', errors='ignore').strip())
                if parsed: frames.append(parsed)
                
        if hardware_failed:
            break
            
        # If we successfully got 3 frames, average and save them instantly
        avg_raw = [sum(col)/len(col) for col in zip(*frames)]
        final_data = normalize_frame(avg_raw)
        
        # This instantly saves the single row to the hard drive
        with open(CSV_FILENAME, mode='a', newline='') as file:
            csv.writer(file).writerow([label] + final_data)
            
        print(f"   Saved sample {i+1}/{SAMPLES_PER_BATCH}")
        time.sleep(0.05)
        
    if not hardware_failed:
        print(f"✅ Success! Saved {SAMPLES_PER_BATCH} rows for '{label}'.")