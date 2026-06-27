import csv
import json
import os
import sys
import threading
import time
from pathlib import Path


# --- CONFIGURATION ---

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.glove_io import DEFAULT_GLOVE_TCP_PORT, TCPLineReceiver, parse_glove_line

CSV_FILENAME = str(ROOT / "data" / "raw" / "sign_language_dataset.csv")
CALIBRATION_FILE = str(ROOT / "sensor_calibration.json")
TCP_HOST = "0.0.0.0"
TCP_PORT = DEFAULT_GLOVE_TCP_PORT


def parse_line(line):
    return parse_glove_line(line)


def open_glove_receiver():
    """Start the same TCP glove receiver used by live_translator.py."""
    print(f"Starting Wi-Fi glove receiver on TCP port {TCP_PORT}...")
    try:
        receiver = TCPLineReceiver(host=TCP_HOST, port=TCP_PORT)
        receiver.start()
    except OSError as exc:
        print(f"Cannot listen on TCP port {TCP_PORT}: {exc}")
        return None
    time.sleep(0.5)
    receiver.reset_input_buffer()
    print(f"Listening for glove frames on {TCP_HOST}:{TCP_PORT}")
    return receiver

# --- SESSION CONFIG ---
COLLECTION_TARGETS = [
    "good morning",
    "hello",
    "i",
    "happy",
    "teacher",
    "thank you",
]
RETAKE_DATASET = True
SAMPLES_PER_LABEL = 10
FRAMES_PER_SAMPLE = 100
FRAME_CAPTURE_TIMEOUT_SEC = 10.0

CALIBRATION_FRAMES = 30
HARDWARE_TIMEOUT_SEC = 15.0

ser = open_glove_receiver()
if not ser:
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
if RETAKE_DATASET:
    if os.path.exists(CSV_FILENAME) and os.path.getsize(CSV_FILENAME) > 0:
        backup_path = f"{CSV_FILENAME}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
        os.replace(CSV_FILENAME, backup_path)
        print(f"📦 Previous word dataset backed up to: {backup_path}")
    with open(CSV_FILENAME, mode='w', newline='') as f:
        csv.writer(f).writerow(HEADERS)
    print("📝 Started a fresh word dataset file with headers.")
elif not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0:
    with open(CSV_FILENAME, mode='w', newline='') as f:
        csv.writer(f).writerow(HEADERS)
    print("📝 Created new dataset file with headers.")
else:
    print("📂 Appending to existing dataset file.")

def wait_for_hardware(timeout_sec=HARDWARE_TIMEOUT_SEC):
    """Confirm the ESP32 is streaming parseable frames before calibration."""
    print("⏳ Waiting for glove data stream...", end="", flush=True)
    deadline = time.time() + timeout_sec
    last_raw = ""
    while time.time() < deadline:
        line = ser.readline(timeout=0.5)
        if not line:
            print(".", end="", flush=True)
            continue
        last_raw = line.strip()
        if parse_line(last_raw):
            print(" OK")
            return True
    print(" FAILED")
    print("\n⚠️ No valid glove frames received.")
    if last_raw:
        print(f"   Last raw line: {last_raw[:120]}")
    print("   Check: both gloves powered on, Glove A paired to Glove B (ESP-NOW), USB cable on Glove B.")
    return False


def collect_calibration_frames(step_name, target_count=CALIBRATION_FRAMES, timeout_sec=HARDWARE_TIMEOUT_SEC):
    """Collect frames with visible progress and a hard timeout."""
    frames = []
    bad_lines = 0
    deadline = time.time() + timeout_sec
    print(f"   Collecting {step_name} calibration ({target_count} frames)...")

    while len(frames) < target_count:
        if time.time() > deadline:
            print(f"\n⚠️ Timeout: only collected {len(frames)}/{target_count} frames for {step_name}.")
            if bad_lines:
                print(f"   Skipped {bad_lines} unparseable line(s).")
            return None

        line = ser.readline(timeout=0.5)
        if not line:
            continue

        parsed = parse_line(line)
        if parsed:
            frames.append(parsed)
            if len(frames) == 1 or len(frames) % 5 == 0 or len(frames) == target_count:
                print(f"   ... {len(frames)}/{target_count}", flush=True)
        else:
            bad_lines += 1

    print(f"   ✅ {step_name} calibration complete.")
    return frames

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
    if not wait_for_hardware():
        ser.close()
        exit(1)

    input("1. Hold both hands completely FLAT and open. Press ENTER...")
    ser.reset_input_buffer()
    flat_frames = collect_calibration_frames("flat/open hand")
    if not flat_frames:
        ser.close()
        exit(1)
    baseline_min = [sum(col) / len(col) for col in zip(*flat_frames)]

    input("2. Close both hands into a TIGHT FIST. Press ENTER...")
    ser.reset_input_buffer()
    fist_frames = collect_calibration_frames("closed fist")
    if not fist_frames:
        ser.close()
        exit(1)
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

def record_glove_background():
    global last_hardware_signal
    while True:
        try:
            line = ser.readline(timeout=0.1)
            if not line:
                continue
            line = line.strip()
            last_hardware_signal = time.time()
            if is_recording:
                parsed = parse_line(line)
                if parsed:
                    with frame_lock:
                        raw_frames.append((time.monotonic(), parsed))
        except Exception:
            pass

listener_thread = threading.Thread(target=record_glove_background, daemon=True)
listener_thread.start()


def record_frames(target_count=FRAMES_PER_SAMPLE, timeout_sec=FRAME_CAPTURE_TIMEOUT_SEC):
    """Record continuously until target_count frames are captured."""
    global is_recording
    with frame_lock:
        raw_frames.clear()
    start = time.monotonic()
    deadline = start + timeout_sec
    is_recording = True
    print(f"   🔴 RECORDING {target_count} frames — perform the sign now!")
    while time.monotonic() < deadline:
        with frame_lock:
            if len(raw_frames) >= target_count:
                break
        time.sleep(0.001)
    is_recording = False
    with frame_lock:
        captured = list(raw_frames[:target_count])
    timed_frames = []
    for frame_time, frame in captured:
        ts = frame_time - start
        if ts < 0:
            ts = 0.0
        timed_frames.append((ts, frame))
    return timed_frames


def save_take(label, timed_frames):
    rows_written = 0
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        for ts, frame in timed_frames:
            normalized = normalize_frame(frame)
            writer.writerow([round(ts, 3), label] + normalized)
            rows_written += 1
        file.flush()
        os.fsync(file.fileno())
    return rows_written


# --- GUIDED RECORDING LOOP ---
print("\n--- GUIDED SIGN DATA COLLECTOR ---")
print(f"Recording: {FRAMES_PER_SAMPLE} frames per sample")
print(f"Samples per sign: {SAMPLES_PER_LABEL}")
print("Press ENTER after each sample to record. Type 'quit' to stop early.\n")
for i, word in enumerate(COLLECTION_TARGETS, 1):
    print(f"  {i:2}. {word}")

session_quit = False
for label_idx, label in enumerate(COLLECTION_TARGETS, 1):
    print(f"\n{'=' * 60}")
    print(f"Sign {label_idx}/{len(COLLECTION_TARGETS)}: '{label}'")
    print(f"Collect {SAMPLES_PER_LABEL} samples ({FRAMES_PER_SAMPLE} frames each)")
    print(f"{'=' * 60}")

    sample_num = 1
    while sample_num <= SAMPLES_PER_LABEL:
        cmd = input(
            f"\n[{label}] Sample {sample_num}/{SAMPLES_PER_LABEL} — "
            f"hold the sign, then press ENTER to record (or 'quit'): "
        ).strip().lower()
        if cmd == 'quit':
            session_quit = True
            break

        if time.time() - last_hardware_signal > 3.0:
            print("\n⚠️ ERROR: Hardware timeout — ESP32 not sending data.")
            session_quit = True
            break

        captured = record_frames()
        total = len(captured)
        print(f"   Captured {total}/{FRAMES_PER_SAMPLE} frames.")

        if total < FRAMES_PER_SAMPLE:
            if total == 0:
                print("   ⚠️ No data captured. Repeat this sample.")
            else:
                print(f"   ⚠️ Only got {total} frames. Repeat this sample.")
            continue

        try:
            rows_written = save_take(label, captured)
            print(f"   ✅ Sample {sample_num} saved — {rows_written} rows written.")
            sample_num += 1
        except Exception as e:
            print(f"   ❌ Save error: {e}")

    if session_quit:
        break

print("\nDone. Dataset saved to:", ABSOLUTE_CSV_PATH)
ser.close()
