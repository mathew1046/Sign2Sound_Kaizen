import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import SignTransformer
from paths import DEFAULT_WEIGHTS, HAND_LANDMARKER


def setup_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    return engine


def speak(engine, text):
    engine.say(text)
    engine.runAndWait()


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


def run_inference(weights_path=DEFAULT_WEIGHTS, confidence_threshold=0.85):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    model = SignTransformer(input_dim=126, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    landmarker = get_hand_landmarker()
    cap = cv2.VideoCapture(0)

    frame_queue = deque(maxlen=30)
    word_buffer = ""
    tts_engine = setup_tts()
    last_pred_time = time.time()

    while cap.isOpened():
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

        frame_queue.append(frame_landmarks)

        if len(frame_queue) == 30 and (time.time() - last_pred_time > 0.16):
            data = np.array(frame_queue).reshape(30, 126)

            mean = data.mean()
            std = data.std()
            if std > 1e-6:
                data = (data - mean) / std

            x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                max_prob, pred_idx = torch.max(probs, 1)

                predicted_class = classes[pred_idx.item()]
                confidence = max_prob.item()

                if confidence > confidence_threshold:
                    print(f"Predicted: {predicted_class} ({confidence:.2f})")
                    word_buffer += predicted_class
                    frame_queue.clear()

            last_pred_time = time.time()

        cv2.putText(
            frame,
            f"Word: {word_buffer}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Sign2Sound Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and word_buffer:
            print(f"Speaking: {word_buffer}")
            speak(tts_engine, word_buffer)
            word_buffer = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ISL alphabet inference")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.85)
    args = parser.parse_args()
    run_inference(weights_path=args.weights, confidence_threshold=args.confidence)
