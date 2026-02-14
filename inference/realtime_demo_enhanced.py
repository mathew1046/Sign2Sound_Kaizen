"""
Enhanced Real-time Sign Language Recognition Demo

This script implements a robust real-time sign language recognition system with:
1. Wrist-relative landmark normalization (matches training preprocessing)
2. Temporal smoothing using One Euro Filter (reduces MediaPipe jitter)
3. Sliding window buffer for sequence-based inference
4. Confidence thresholding for better sign detection

Usage:
    python inference/realtime_demo_enhanced.py \
        --model checkpoints/best_model.pth \
        --device cuda \
        --confidence 0.8 \
        --window_size 30

Author: Team Kaizen  
Date: January 2026
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import load_model
from features.hand_landmarks import HandLandmarkDetector
from inference.tts import TextToSpeech
from inference.feature_processing import (
    normalize_landmarks_wrist_relative,
    LandmarkSmoother,
    SlidingWindowBuffer,
    validate_features,
    pad_or_truncate
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_class_mapping(csv_path: str) -> Dict[int, str]:
    """Load class mapping from CSV file."""
    import pandas as pd
    
    if not Path(csv_path).exists():
        logger.warning(f"Class mapping file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['class_idx'], df['class_name']))
    return mapping


class EnhancedRealtimeDemoApp:
    """
    Enhanced real-time sign language recognition with robust preprocessing.
    
    Key improvements:
    - Wrist-relative normalization (training-aligned)
    - One Euro Filter for jitter reduction
    - Sliding window buffer for temporal consistency
    - Confidence thresholding
    """
    
    # Default ISL class mapping (A-Z excluding R)
    DEFAULT_CLASS_MAPPING = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
        8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
        15: "P", 16: "Q", 17: "S", 18: "T", 19: "U", 20: "V", 21: "W",
        22: "X", 23: "Y", 24: "Z"
    }
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.8,
        window_size: int = 30,
        use_tts: bool = True,
        use_smoothing: bool = True
    ):
        """
        Initialize enhanced demo app.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for sign detection
            window_size: Sliding window size for sequences
            use_tts: Enable text-to-speech
            use_smoothing: Enable One Euro Filter smoothing
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        
        # Load model (automatically detects CTC vs standard)
        self.model, self.checkpoint = load_model(model_path, device=str(self.device))
        logger.info(f"Model loaded: {self.checkpoint.get('model_type', 'unknown')}")
        
        # Initialize hand detector (video mode for better tracking)
        self.hand_detector = HandLandmarkDetector(
            static_image_mode=False,  # Video mode
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load class mapping
        csv_path = PROJECT_ROOT / "data" / "processed" / "class_mapping.csv"
        loaded_mapping = load_class_mapping(str(csv_path))
        
        if loaded_mapping:
            # Extract letter from "ISL_X" format
            self.class_mapping = {}
            for idx, name in loaded_mapping.items():
                if name.startswith("ISL_"):
                    self.class_mapping[idx] = name.replace("ISL_", "")
            logger.info(f"Loaded ISL class mapping: {len(self.class_mapping)} classes")
        else:
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logger.warning("Using default class mapping")
        
        # Sliding window buffer
        self.window_size = window_size
        self.sliding_buffer = SlidingWindowBuffer(window_size=window_size, feature_dim=126)
        logger.info(f"Sliding window size: {window_size} frames")
        
        # Temporal smoothing
        self.use_smoothing = use_smoothing
        if use_smoothing:
            self.landmark_smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.007)
            logger.info("One Euro Filter smoothing enabled")
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Prediction smoothing (for display stability)
        self.prediction_history = deque(maxlen=5)
        
        # TTS
        self.use_tts = use_tts
        if use_tts:
            self.tts = TextToSpeech()
            logger.info("TTS enabled")
        
        # State
        self.accumulated_text = ""
        self.last_spoken = ""
        self.frame_count = 0
        
        logger.info(f"Enhanced demo initialized on {self.device}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float, str]:
        """
        Process video frame with enhanced pipeline.
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Tuple of (annotated_frame, prediction, confidence, status_message)
        """
        self.frame_count += 1
        
        # Detect hands
        hand_landmarks_list = self.hand_detector.detect_hands(frame)
        
        # Draw landmarks for visualization
        annotated_frame = frame.copy()
        if hand_landmarks_list is not None:
            annotated_frame = self.hand_detector.draw_landmarks(annotated_frame, hand_landmarks_list)
        
        # Check if hands detected
        if hand_landmarks_list is None:
            status = "No hands detected"
            self._draw_status(annotated_frame, status, "", 0.0, (0, 0, 255))
            return annotated_frame, "", 0.0, status
        
        # Extract raw features
        raw_features = self.hand_detector.process_detected_landmarks(hand_landmarks_list)
        
        if raw_features is None or not validate_features(raw_features):
            status = "Invalid features"
            self._draw_status(annotated_frame, status, "", 0.0, (0, 0, 255))
            return annotated_frame, "", 0.0, status
        
        # Step 1: Apply temporal smoothing (reduces MediaPipe jitter)
        if self.use_smoothing:
            features = self.landmark_smoother.smooth(raw_features, timestamp=time.time())
        else:
            features = raw_features
        
        # Step 2: Apply wrist-relative normalization (matches training preprocessing)
        features = normalize_landmarks_wrist_relative(features)
        
        # Step 3: Add to sliding window buffer
        self.sliding_buffer.add(features)
        
        # Step 4: Check if buffer is ready for inference
        if not self.sliding_buffer.is_ready():
            status = f"Buffering... ({len(self.sliding_buffer)}/{self.window_size})"
            self._draw_status(annotated_frame, status, "", 0.0, (255, 165, 0))
            return annotated_frame, "", 0.0, status
        
        # Step 5: Get sequence from buffer
        sequence = self.sliding_buffer.get_sequence()
        
        # Step 6: Pad to model's expected length (60 frames)
        sequence_padded = pad_or_truncate(sequence, max_len=60)
        
        # Step 7: Run inference
        prediction, confidence = self._predict(sequence_padded)
        
        # Step 8: Get class name
        letter = self.class_mapping.get(prediction, f"Class_{prediction}")
        
        # Step 9: Apply confidence thresholding (but still show prediction)
        if confidence < self.confidence_threshold:
            status = f"Low Confidence ({confidence:.2f})"
            color = (255, 165, 0)  # Orange for low confidence
            self._draw_status(annotated_frame, status, letter, confidence, color)
            return annotated_frame, letter, confidence, status
        
        # Step 10: Display with high confidence
        status = "Detected"
        color = (0, 255, 0)  # Green for confident detection
        self._draw_status(annotated_frame, status, letter, confidence, color)
        
        # Update accumulated text (only for high confidence)
        self._update_accumulated_text(letter, confidence)
        
        return annotated_frame, letter, confidence, status
    
    def _predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Run model inference on sequence.
        
        Args:
            sequence: Input sequence of shape (60, 126)
        
        Returns:
            Tuple of (class_id, confidence)
        """
        # Prepare tensors
        features_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
        seq_length = torch.tensor([self.window_size]).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(features_tensor, seq_length)
            
            # Handle both CTC and standard models
            if logits.dim() == 3:  # CTC: (T, B, C)
                logits = logits.mean(dim=0)  # (B, C)
            
            probs = torch.softmax(logits, dim=1)
            class_id = torch.argmax(logits, dim=1).item()
            confidence = probs[0, class_id].item()
        
        # Add to prediction history for stability
        self.prediction_history.append(class_id)
        
        # Use mode of recent predictions for display stability
        if len(self.prediction_history) >= 3:
            unique, counts = np.unique(list(self.prediction_history), return_counts=True)
            stable_pred = unique[np.argmax(counts)]
            return int(stable_pred), confidence
        
        return int(class_id), confidence
    
    def _draw_status(self, frame: np.ndarray, status: str, letter: str, 
                     confidence: float, color: Tuple[int, int, int]):
        """Draw status information on frame."""
        h, w = frame.shape[:2]
        
        # Status message (top-left)
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Detected letter (center, large)
        if letter:
            font_scale = 3.0
            thickness = 5
            text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Draw background rectangle
            padding = 20
            cv2.rectangle(frame,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)
            
            # Draw letter
            cv2.putText(frame, letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Draw confidence bar
            bar_width = 200
            bar_height = 20
            bar_x = (w - bar_width) // 2
            bar_y = text_y + 50
            
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            
            filled_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + filled_width, bar_y + bar_height),
                         color, -1)
            
            cv2.putText(frame, f"{confidence:.2f}", 
                       (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Accumulated text (top-right)
        if self.accumulated_text:
            cv2.putText(frame, f"Text: {self.accumulated_text}", 
                       (w - 400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame count and buffer status (bottom-left)
        buffer_status = f"Buffer: {len(self.sliding_buffer)}/{self.window_size}"
        cv2.putText(frame, buffer_status, (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_accumulated_text(self, letter: str, confidence: float):
        """Update accumulated text with new detection."""
        if confidence >= self.confidence_threshold and letter:
            # Add letter if it's different from last character
            if not self.accumulated_text or self.accumulated_text[-1] != letter:
                self.accumulated_text += letter
                
                # Speak if TTS enabled and different from last spoken
                if self.use_tts and letter != self.last_spoken:
                    self.tts.speak(letter)
                    self.last_spoken = letter
    
    def clear_text(self):
        """Clear accumulated text."""
        self.accumulated_text = ""
        self.last_spoken = ""
    
    def reset_buffer(self):
        """Reset sliding window buffer."""
        self.sliding_buffer.clear()
        if self.use_smoothing:
            self.landmark_smoother.reset()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Real-time Sign Language Recognition")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Sliding window size (frames)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--no_tts', action='store_true',
                       help='Disable text-to-speech')
    parser.add_argument('--no_smoothing', action='store_true',
                       help='Disable temporal smoothing')
    
    args = parser.parse_args()
    
    # Initialize app
    app = EnhancedRealtimeDemoApp(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        window_size=args.window_size,
        use_tts=not args.no_tts,
        use_smoothing=not args.no_smoothing
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {args.camera}")
        return
    
    # Force 720p @ 30 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera settings: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")

    logger.info("Starting real-time demo. Press 'q' to quit, 'c' to clear text, 'r' to reset buffer")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Process frame
            annotated_frame, letter, confidence, status = app.process_frame(frame)
            
            # Display
            cv2.imshow("Enhanced Sign Language Recognition", annotated_frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                app.clear_text()
                logger.info("Text cleared")
            elif key == ord('r'):
                app.reset_buffer()
                logger.info("Buffer reset")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Demo stopped")


if __name__ == "__main__":
    main()
