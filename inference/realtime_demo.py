"""
Real-time Sign Language Recognition Demo

Real-time webcam-based sign language recognition with visualization and smoothing.

Usage:
    python inference/realtime_demo.py --model checkpoints/best_model.pth

Controls:
    SPACE: Capture prediction and add to sentence
    C: Clear accumulated text
    Q: Quit

Author: Team Kaizen
Date: January 2026
"""

import argparse
import logging
import sys
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np
import cv2
import torch

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import load_model
from features.hand_landmarks import HandLandmarkDetector
from features.feature_utils import pad_or_truncate
from inference.utils import load_class_mapping
from inference.tts import TextToSpeech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeDemoApp:
    """Real-time sign language recognition demo."""

    # Fallback class mapping if CSV not found
    DEFAULT_CLASS_MAPPING = {
        # ISL (0-24) - A-Z excluding R
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "S",
        18: "T",
        19: "U",
        20: "V",
        21: "W",
        22: "X",
        23: "Y",
        24: "Z",
    }

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize demo app.

        Args:
            model_path: Path to model checkpoint
            device: Device to use
            smoothing_window: Number of frames for prediction smoothing
            confidence_threshold: Minimum confidence threshold
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Load model
        self.model, self.checkpoint = load_model(model_path, device=str(self.device))

        # Load class mapping from CSV
        csv_path = PROJECT_ROOT / "data" / "processed" / "class_mapping.csv"
        loaded_mapping = load_class_mapping(str(csv_path))

        if loaded_mapping:
            # Extract just the letter from "ISL_X" format
            self.class_mapping = {}
            for idx, name in loaded_mapping.items():
                if name.startswith("ISL_"):
                    # Extract ISL letter after "ISL_"
                    self.class_mapping[idx] = name.replace("ISL_", "")
                else:
                    self.class_mapping[idx] = name
            logger.info(
                f"Loaded class mapping from CSV with {len(self.class_mapping)} classes"
            )
        else:
            # Fallback to default mapping
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logger.warning("Using default class mapping")

        # Initialize extractor
        self.extractor = HandLandmarkDetector(
            static_image_mode=False,  # Video mode
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        # Initialize TTS
        self.tts = TextToSpeech()

        # Temporal consensus filter
        self.smoothing_window = smoothing_window
        self.consensus_buffer: Deque[int] = deque(maxlen=10)
        self.consensus_required = 8
        self.confidence_threshold = confidence_threshold
        self.stable_class_id: Optional[int] = None
        self.stable_letter: str = "..."
        self.class_thresholds = {
            "A": 0.90,
            "M": 0.90,
            "N": 0.90,
        }
        self.default_threshold = 0.75

        # Accumulator for text
        self.accumulated_text = ""

        logger.info(f"Demo app initialized on {self.device}")

    def get_consensus_prediction(self, current_pred: int) -> Tuple[int, float, bool]:
        """
        Get consensus prediction using temporal buffer.

        Args:
            current_pred: Current frame prediction

        Returns:
            Tuple of (consensus_class_id, consensus_ratio, has_consensus)
        """
        self.consensus_buffer.append(current_pred)

        if len(self.consensus_buffer) == 0:
            return -1, 0.0, False

        unique, counts = np.unique(list(self.consensus_buffer), return_counts=True)
        top_idx = int(np.argmax(counts))
        consensus_id = int(unique[top_idx])
        consensus_ratio = float(np.max(counts) / len(self.consensus_buffer))
        has_consensus = counts[top_idx] >= self.consensus_required

        return consensus_id, consensus_ratio, has_consensus

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, float, str]:
        """
        Process video frame and get prediction.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (annotated_frame, class_id, confidence, class_name)
        """
        # Detect hands once to enable both features and visualization
        hand_landmarks_list = self.extractor.detect_hands(frame)

        if hand_landmarks_list is None:
            cv2.putText(
                frame,
                "No hands detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return frame, -1, 0.0, "NO_HANDS"

        # Draw skeletal keypoints for better visualization
        annotated_frame = self.extractor.draw_landmarks(frame, hand_landmarks_list)

        # Extract features from detected landmarks
        features = self.extractor.process_detected_landmarks(hand_landmarks_list)

        if features is None:
            cv2.putText(
                annotated_frame,
                "Landmark processing failed",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return annotated_frame, -1, 0.0, "NO_HANDS"

        # Pad features
        features = pad_or_truncate(features, max_len=60)
        features_tensor = (
            torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        )
        seq_length = torch.tensor([60]).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor, seq_length)
            
            # Handle both model types
            # CTC model outputs (seq_len, batch, classes)
            # Standard model outputs (batch, classes)
            if logits.dim() == 3:  # CTC output: (T, B, C)
                # Use mean across timesteps for prediction
                logits = logits.mean(dim=0)  # (B, C)
            
            probs = torch.softmax(logits, dim=1)
            class_id_tensor = torch.argmax(logits, dim=1)
            class_id: int = int(class_id_tensor.item())
            confidence = float(probs[0][class_id].item())

        # Apply class-specific confidence thresholds
        current_letter = self.class_mapping.get(class_id, f"Class_{class_id}")
        required_conf = self.class_thresholds.get(current_letter, self.default_threshold)
        gated_class_id = class_id if confidence >= required_conf else -1

        if gated_class_id == -1:
            logger.debug(
                "Low confidence for class %s (%.3f < %.3f)",
                current_letter,
                confidence,
                required_conf,
            )

        # Apply temporal consensus filter
        consensus_id, consensus_ratio, has_consensus = self.get_consensus_prediction(
            gated_class_id
        )

        if has_consensus and consensus_id != -1:
            self.stable_class_id = consensus_id
            self.stable_letter = self.class_mapping.get(
                consensus_id, f"Class_{consensus_id}"
            )
        else:
            if self.stable_class_id is None:
                self.stable_letter = "..."

        logger.info(
            "Frame prediction: current=%s (%.2f%%, thresh=%.2f), gated=%s, "
            "consensus=%s (%.2f%%), stable=%s",
            current_letter,
            confidence * 100.0,
            required_conf,
            "none" if gated_class_id == -1 else str(gated_class_id),
            "none" if consensus_id == -1 else str(consensus_id),
            consensus_ratio * 100.0,
            self.stable_letter,
        )

        # Annotate frame
        color = (0, 255, 0) if has_consensus else (0, 0, 255)

        # Display letter prominently
        cv2.putText(
            annotated_frame,
            f"Sign: {self.stable_letter}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3,
        )
        cv2.putText(
            annotated_frame,
            f"Consensus: {consensus_ratio * 100:.1f}% | Current: {current_letter} ({confidence * 100:.1f}%)",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        if self.accumulated_text:
            cv2.putText(
                annotated_frame,
                f"Sentence: {self.accumulated_text}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Instructions
        cv2.putText(
            annotated_frame,
            "SPACE: Add | C: Clear | Q: Quit",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        return annotated_frame, consensus_id, consensus_ratio, self.stable_letter

    def run(self, camera_id: int = 0, fps: int = 30):
        """
        Run real-time demo.

        Args:
            camera_id: Camera device ID (default: 0)
            fps: Target frames per second
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error("Failed to open camera")
            return

        logger.info("Starting real-time demo. Press 'q' to quit.")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Process frame
            annotated, class_id, confidence, letter = self.process_frame(frame)

            # Display
            cv2.imshow("Sign Language Recognition", annotated)

            # Handle key input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quitting...")
                break
            elif key == ord(" "):  # Space bar
                if self.stable_letter != "...":
                    self.accumulated_text += self.stable_letter
                    logger.info(f"Added '{self.stable_letter}' to sentence")
                    # Optionally speak the word
                    # self.tts.speak(class_name)
            elif key == ord("c"):  # Clear
                self.accumulated_text = ""
                self.consensus_buffer.clear()
                self.stable_class_id = None
                self.stable_letter = "..."
                logger.info("Cleared predictions")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        self.tts.close()

        logger.info(f"Demo ended. Processed {frame_count} frames.")

    def close(self):
        """Clean up resources."""
        self.extractor.close()
        self.tts.close()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time sign language recognition demo"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda or cpu)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument(
        "--smoothing", type=int, default=5, help="Prediction smoothing window size"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="Confidence threshold"
    )

    args = parser.parse_args()

    # Run demo
    app = RealtimeDemoApp(
        args.model,
        device=args.device,
        smoothing_window=args.smoothing,
        confidence_threshold=args.confidence,
    )

    app.run(camera_id=args.camera)
    app.close()


if __name__ == "__main__":
    main()
