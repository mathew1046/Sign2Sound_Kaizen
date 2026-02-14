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
from typing import Deque, Tuple
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
from inference.feature_processing import normalize_landmarks_wrist_relative
from inference.tts import TextToSpeech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeDemoApp:
    """Real-time sign language recognition demo."""
    
    # Fallback class mapping if CSV not found (ISL only)
    DEFAULT_CLASS_MAPPING = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
        22: 'X', 23: 'Y', 24: 'Z'
    }
    
    def __init__(self, model_path: str, device: str = None,
                 smoothing_window: int = 5,
                 confidence_threshold: float = 0.7):
        """
        Initialize demo app.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            smoothing_window: Number of frames for prediction smoothing
            confidence_threshold: Minimum confidence threshold
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        
        # Load model
        self.model, self.checkpoint = load_model(model_path, device=str(self.device))
        
        # Load class mapping from CSV
        csv_path = PROJECT_ROOT / 'data' / 'processed' / 'class_mapping.csv'
        loaded_mapping = load_class_mapping(str(csv_path))
        
        if loaded_mapping:
            # Extract just the letter from "ISL_X" format
            self.class_mapping = {}
            for idx, name in loaded_mapping.items():
                if name.startswith('ISL_'):
                    self.class_mapping[idx] = name.replace('ISL_', '')
            logger.info(f"Loaded ISL class mapping from CSV with {len(self.class_mapping)} classes")
        else:
            # Fallback to default mapping
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logger.warning("Using default class mapping")
        
        # Initialize extractor
        self.extractor = HandLandmarkDetector(
            static_image_mode=False,  # Video mode
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Initialize TTS
        self.tts = TextToSpeech()
        
        # Smoothing
        self.smoothing_window = smoothing_window
        self.prediction_history: Deque[int] = deque(maxlen=smoothing_window)
        self.confidence_threshold = confidence_threshold
        
        # Accumulator for text
        self.accumulated_text = ""
        
        logger.info(f"Demo app initialized on {self.device}")
    
    def get_smooth_prediction(self, current_pred: int) -> Tuple[int, float]:
        """
        Get smoothed prediction using history.
        
        Args:
            current_pred: Current frame prediction
        
        Returns:
            Tuple of (smoothed_class_id, confidence)
        """
        self.prediction_history.append(current_pred)
        
        if len(self.prediction_history) == 0:
            return -1, 0.0
        
        # Mode (most common prediction)
        unique, counts = np.unique(list(self.prediction_history), return_counts=True)
        smoothed_pred = unique[np.argmax(counts)]
        confidence = np.max(counts) / len(self.prediction_history)
        
        return smoothed_pred, confidence
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, float, str]:
        """
        Process video frame and get prediction.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Tuple of (annotated_frame, class_id, confidence, class_name)
        """
        # Extract features
        features = self.extractor.process_image(frame)
        
        if features is None:
            # No hands detected
            cv2.putText(frame, "No hands detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, -1, 0.0, "NO_HANDS"
        
        # Apply wrist-relative normalization (training-aligned)
        features = normalize_landmarks_wrist_relative(features)

        # Pad features
        features = pad_or_truncate(features, max_len=60)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        seq_length = torch.tensor([60]).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor, seq_length)
            probs = torch.softmax(logits, dim=1)
            class_id = torch.argmax(logits, dim=1).item()
            confidence = probs[0, class_id].item()
        
        # Apply smoothing
        smoothed_id, smooth_conf = self.get_smooth_prediction(class_id)
        
        # Get letter/character from class mapping
        letter = self.class_mapping.get(smoothed_id, f"Class_{smoothed_id}")
        
        # Annotate frame
        color = (0, 255, 0) if smooth_conf >= self.confidence_threshold else (0, 0, 255)
        
        # Display letter prominently
        cv2.putText(frame, f"Sign: {letter}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Class {smoothed_id} | Confidence: {smooth_conf*100:.1f}%", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.accumulated_text:
            cv2.putText(frame, f"Sentence: {self.accumulated_text}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "SPACE: Add | C: Clear | Q: Quit", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame, smoothed_id, smooth_conf, letter
    
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
        
        # Force 720p @ 30 FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera settings: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")

        logger.info("Starting real-time demo. Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, class_id, confidence, letter = self.process_frame(frame)
            
            # Display
            cv2.imshow('Sign Language Recognition', annotated)
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting...")
                break
            elif key == ord(' '):  # Space bar
                if confidence >= self.confidence_threshold:
                    self.accumulated_text += letter
                    logger.info(f"Added '{letter}' to sentence")
                    # Optionally speak the word
                    # self.tts.speak(class_name)
            elif key == ord('c'):  # Clear
                self.accumulated_text = ""
                self.prediction_history.clear()
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
    parser = argparse.ArgumentParser(description="Real-time sign language recognition demo")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--smoothing', type=int, default=5,
                       help='Prediction smoothing window size')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Run demo
    app = RealtimeDemoApp(
        args.model,
        device=args.device,
        smoothing_window=args.smoothing,
        confidence_threshold=args.confidence
    )
    
    app.run(camera_id=args.camera)
    app.close()


if __name__ == "__main__":
    main()
