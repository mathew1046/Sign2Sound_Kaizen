"""
Video Inference Module for Sign Language Recognition

Process video files to detect and recognize sign language gestures.

Usage:
    python inference/video_infer.py \
        --model checkpoints/best_model.pth \
        --input path/to/video.mp4 \
        --output results/video_predictions.json

Author: Team Kaizen
Date: January 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import cv2
import torch
from collections import deque
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import load_model
from features.hand_landmarks import HandLandmarkDetector
from features.feature_utils import pad_or_truncate
from inference.utils import load_class_mapping
from inference.feature_processing import (
    normalize_landmarks_wrist_relative,
    LandmarkSmoother,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoSignLanguageRecognizer:
    """Recognize sign language from video files."""
    
    # Default class mapping: class_index -> letter
    DEFAULT_CLASS_MAPPING = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
        7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
        14: 'O', 15: 'P', 16: 'Q', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 
        21: 'W', 22: 'X', 23: 'Y', 24: 'Z'
    }
    
    def __init__(self, model_path: str, device: str = None, window_size: int = 30, 
                 confidence_threshold: float = 0.8):
        """
        Initialize video recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (auto-detect if None)
            window_size: Sliding window size for temporal context (frames)
            confidence_threshold: Minimum confidence for prediction (0.0-1.0)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model, self.checkpoint = load_model(model_path, device=str(self.device))
        self.model.eval()
        
        # Load class mapping from CSV
        csv_path = PROJECT_ROOT / 'data' / 'processed' / 'class_mapping.csv'
        loaded_mapping = load_class_mapping(str(csv_path))
        
        if loaded_mapping:
            # Extract just the letter from "ISL_X" format
            self.class_mapping = {}
            for idx, name in loaded_mapping.items():
                if name.startswith('ISL_'):
                    self.class_mapping[idx] = name.replace('ISL_', '')
                else:
                    self.class_mapping[idx] = name
            logger.info(f"Loaded class mapping from CSV with {len(self.class_mapping)} classes")
        else:
            # Fallback to default mapping
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logger.warning("Using default class mapping")
        
        # Initialize feature extractor
        self.extractor = HandLandmarkDetector(
            static_image_mode=False,  # Video mode for better tracking
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Initialize landmark smoother for jitter reduction
        self.smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.007)
        
        # Sliding window buffer for temporal context
        self.feature_buffer = deque(maxlen=window_size)
        
        logger.info(f"Video recognizer initialized on {self.device}")
        logger.info(f"Window size: {window_size}, Confidence threshold: {confidence_threshold}")
    
    def process_video(self, video_path: str, output_video: str = None, 
                     save_predictions: str = None) -> Tuple[List[Dict], float]:
        """
        Process entire video and recognize signs.
        
        Args:
            video_path: Path to video file
            output_video: Path to save video with predictions (optional)
            save_predictions: Path to save predictions JSON (optional)
        
        Returns:
            Tuple of (predictions list, video FPS)
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return [], 0.0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Duration: {total_frames/fps:.2f}s")
        
        # Setup video writer if output requested
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        predictions = []
        frame_count = 0
        detected_signs = deque(maxlen=5)  # Keep last 5 detections for smoothing
        
        logger.info(f"Processing video...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % max(1, total_frames // 10) == 0:
                    logger.info(f"  Processed {frame_count}/{total_frames} frames")
                
                # Extract landmarks
                landmarks = self.extractor.process_image(frame)
                
                # Add prediction to frame
                frame_display = frame.copy()
                
                if landmarks is not None:
                    # Apply preprocessing
                    features = normalize_landmarks_wrist_relative(landmarks)
                    features = self.smoother.smooth(features)
                    
                    # Add to buffer
                    self.feature_buffer.append(features)
                    detected_signs.append(True)
                else:
                    detected_signs.append(False)
                
                # Make prediction if buffer is ready
                if len(self.feature_buffer) >= min(self.window_size // 2, 10):
                    # Prepare sequence from buffer
                    sequence = np.array(list(self.feature_buffer))
                    sequence = pad_or_truncate(sequence, max_len=60)
                    
                    # Predict
                    class_id, confidence, class_name = self._predict(sequence)
                    
                    # Store prediction
                    pred_entry = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'hands_detected': landmarks is not None
                    }
                    predictions.append(pred_entry)
                    
                    # Draw on frame
                    frame_display = self._draw_prediction(
                        frame_display, class_name, confidence
                    )
                
                # Add frame counter
                cv2.putText(
                    frame_display, f"Frame: {frame_count}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Write output video if requested
                if writer:
                    writer.write(frame_display)
                
        finally:
            cap.release()
            if writer:
                writer.release()
        
        logger.info(f"✓ Video processing complete ({frame_count} frames)")
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions, save_predictions, fps)
        
        return predictions, fps
    
    def _predict(self, sequence: np.ndarray) -> Tuple[int, float, str]:
        """
        Make prediction from feature sequence.
        
        Args:
            sequence: Feature sequence of shape (seq_len, 126)
        
        Returns:
            Tuple of (class_id, confidence, class_name)
        """
        try:
            # Prepare tensor
            features_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
            seq_length = torch.tensor([sequence.shape[0]]).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(features_tensor, seq_length)
                
                # Handle both model types
                if logits.dim() == 3:  # CTC output: (T, B, C)
                    logits = logits.mean(dim=0)  # (B, C)
                
                probs = torch.softmax(logits, dim=1)
                class_id = torch.argmax(logits, dim=1).item()
                confidence = probs[0, class_id].item()
            
            class_name = self.class_mapping.get(class_id, f"Class_{class_id}")
            
            return class_id, confidence, class_name
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return 0, 0.0, "ERROR"
    
    def _draw_prediction(self, frame: np.ndarray, class_name: str, 
                        confidence: float) -> np.ndarray:
        """
        Draw prediction on frame.
        
        Args:
            frame: Input frame
            class_name: Predicted class name
            confidence: Prediction confidence
        
        Returns:
            Frame with drawn prediction
        """
        height, width = frame.shape[:2]
        
        # Determine color based on confidence
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0)  # Green
            status = "Detected"
        else:
            color = (0, 165, 255)  # Orange
            status = "Low Confidence"
        
        # Draw background box
        box_width = 400
        box_height = 80
        x = width - box_width - 10
        y = height - box_height - 10
        
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), 
                     color, 2)
        
        # Draw text
        cv2.putText(
            frame, f"Sign: {class_name}",
            (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, color, 2
        )
        cv2.putText(
            frame, f"Confidence: {confidence*100:.1f}%",
            (x + 10, y + 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )
        
        return frame
    
    def _save_predictions(self, predictions: List[Dict], output_path: str, fps: float):
        """
        Save predictions to JSON file.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save JSON
            fps: Video frames per second
        """
        output_data = {
            'video_fps': fps,
            'total_frames': max([p['frame'] for p in predictions]) if predictions else 0,
            'total_predictions': len(predictions),
            'predictions': predictions,
            'summary': self._generate_summary(predictions)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ Predictions saved to {output_path}")
    
    def _generate_summary(self, predictions: List[Dict]) -> Dict:
        """
        Generate summary statistics from predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            Summary dictionary
        """
        if not predictions:
            return {}
        
        # Count class occurrences
        class_counts = {}
        high_conf_count = 0
        avg_confidence = 0.0
        
        for pred in predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if pred['confidence'] >= self.confidence_threshold:
                high_conf_count += 1
            
            avg_confidence += pred['confidence']
        
        avg_confidence /= len(predictions)
        
        # Sort classes by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_predictions': len(predictions),
            'unique_classes': len(class_counts),
            'high_confidence_predictions': high_conf_count,
            'average_confidence': float(avg_confidence),
            'most_frequent_signs': sorted_classes[:5],
            'class_distribution': dict(sorted_classes)
        }
    
    def close(self):
        """Release resources."""
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description="Recognize sign language in video")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Path to save output video with predictions')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save predictions as JSON')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Minimum confidence for high-confidence prediction (0.0-1.0)')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Sliding window size for temporal context (frames)')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {args.input}")
        return
    
    # Initialize recognizer
    recognizer = VideoSignLanguageRecognizer(
        args.model,
        device=args.device,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process video
    predictions, fps = recognizer.process_video(
        args.input,
        output_video=args.output_video,
        save_predictions=args.output_json
    )
    
    # Print results summary
    logger.info("="*70)
    logger.info("VIDEO PROCESSING SUMMARY")
    logger.info("="*70)
    logger.info(f"Total predictions: {len(predictions)}")
    
    if predictions:
        # Count detections by class
        class_counts = {}
        high_conf_count = 0
        
        for pred in predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if pred['confidence'] >= args.confidence_threshold:
                high_conf_count += 1
        
        logger.info(f"High confidence detections: {high_conf_count}/{len(predictions)}")
        logger.info(f"Unique classes detected: {len(class_counts)}")
        
        # Show top 5 classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top detected signs:")
        for sign, count in sorted_classes[:5]:
            pct = (count / len(predictions)) * 100
            logger.info(f"  {sign}: {count} times ({pct:.1f}%)")
    
    logger.info("="*70)
    
    recognizer.close()


if __name__ == "__main__":
    main()
