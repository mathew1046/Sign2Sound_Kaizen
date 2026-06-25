import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from paths import HAND_LANDMARKER


def get_hand_landmarker():
    if not HAND_LANDMARKER.exists():
        print("Downloading hand_landmarker.task...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, HAND_LANDMARKER)

    base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def record_class_r(output_dir, num_sequences=120, frames_per_seq=30):
    os.makedirs(output_dir, exist_ok=True)
    landmarker = get_hand_landmarker()
    cap = cv2.VideoCapture(0)

    print("Starting in 3 seconds...")
    time.sleep(3)

    for seq in range(num_sequences):
        print(f"Recording sequence {seq + 1}/{num_sequences}...")
        sequence_data = []

        for frame_num in range(frames_per_seq):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)

            frame_landmarks = np.zeros((42, 3))
            if results.hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.hand_landmarks):
                    hand_label = results.handedness[hand_idx][0].category_name
                    offset = 0 if hand_label == "Left" else 21
                    for i, lm in enumerate(hand_lms):
                        frame_landmarks[offset + i] = [lm.x, lm.y, lm.z]

            sequence_data.append(frame_landmarks)

            cv2.putText(
                frame,
                f"Seq: {seq + 1} Frame: {frame_num + 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        np.save(os.path.join(output_dir, f"{seq:03d}.npy"), np.array(sequence_data))
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Record ISL 'R' sign sequences from webcam")
    parser.add_argument("--output-dir", required=True, help="Directory to save .npy sequences")
    parser.add_argument("--num-sequences", type=int, default=120)
    parser.add_argument("--frames-per-seq", type=int, default=30)
    args = parser.parse_args()

    record_class_r(args.output_dir, args.num_sequences, args.frames_per_seq)


if __name__ == "__main__":
    main()
