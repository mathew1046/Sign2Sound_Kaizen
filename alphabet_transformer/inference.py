import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_crop import (
    default_cansik_paths,
    extract_frame_landmarks,
    try_create_cansik_detector,
)
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


def run_inference(
    weights_path=DEFAULT_WEIGHTS,
    confidence_threshold=0.85,
    *,
    use_crop=True,
    crop_pad=0.15,
    hand_det_confidence=0.2,
    cansik_cfg=None,
    cansik_weights=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    model = SignTransformer(input_dim=126, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    landmarker = get_hand_landmarker()
    cfg_path, weights_path_cansik = cansik_cfg or default_cansik_paths()[0], cansik_weights or default_cansik_paths()[1]
    detector = try_create_cansik_detector(cfg_path, weights_path_cansik, confidence=hand_det_confidence) if use_crop else None
    if use_crop and detector is not None:
        print(f"[inference] hand crop enabled (pad={crop_pad})")
    elif use_crop:
        print("[inference] hand crop requested but detector unavailable — full-frame fallback")

    cap = cv2.VideoCapture(0)

    frame_queue = deque(maxlen=30)
    word_buffer = ""
    tts_engine = setup_tts()
    last_pred_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_landmarks = extract_frame_landmarks(
            landmarker,
            frame,
            use_crop=use_crop,
            detector=detector,
            pad_frac=crop_pad,
        )
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

    default_cfg, default_weights = default_cansik_paths()
    parser = argparse.ArgumentParser(description="Real-time ISL alphabet inference")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop hands with Cansik before MediaPipe (default: on)",
    )
    parser.add_argument("--crop-pad", type=float, default=0.15, help="Padding fraction around hand union bbox")
    parser.add_argument("--hand-det-confidence", type=float, default=0.2, help="Cansik detection threshold")
    parser.add_argument("--cansik-cfg", type=Path, default=default_cfg)
    parser.add_argument("--cansik-weights", type=Path, default=default_weights)
    args = parser.parse_args()
    run_inference(
        weights_path=args.weights,
        confidence_threshold=args.confidence,
        use_crop=args.crop,
        crop_pad=args.crop_pad,
        hand_det_confidence=args.hand_det_confidence,
        cansik_cfg=args.cansik_cfg,
        cansik_weights=args.cansik_weights,
    )
