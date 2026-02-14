"""
Inference Module for Sign Language Recognition

Single image/batch prediction using trained BiLSTM model.

Usage:
    python inference/infer.py --model checkpoints/best_model.pth --input path/to/image.jpg

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
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import load_model
from features.hand_landmarks import HandLandmarkDetector
from features.feature_utils import pad_or_truncate
from inference.utils import load_class_mapping
from inference.feature_processing import normalize_landmarks_wrist_relative

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignLanguagePredictor:
    """Predict sign language from images."""
    
    # ISL-only class mapping: class_index -> letter (A-Z excluding R)
    DEFAULT_CLASS_MAPPING = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
        22: 'X', 23: 'Y', 24: 'Z'
    }
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (auto-detect if None)
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
        
        # Initialize feature extractor
        self.extractor = HandLandmarkDetector(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        logger.info(f"Predictor initialized on {self.device}")
    
    def predict_image(self, image_path: str) -> Tuple[int, float, str]:
        """
        Predict from single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (class_id, confidence, class_name)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, 0.0, "ERROR"
        
        # Extract features
        features = self.extractor.process_image(image)
        if features is None:
            logger.warning(f"Failed to extract features from {image_path}")
            return None, 0.0, "NO_HANDS"
        
        # Apply wrist-relative normalization (training-aligned)
        features = normalize_landmarks_wrist_relative(features)

        # Prepare input
        features = pad_or_truncate(features, max_len=60)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        seq_length = torch.tensor([1]).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor, seq_length)
            probs = torch.softmax(logits, dim=1)
            class_id = torch.argmax(logits, dim=1).item()
            confidence = probs[0, class_id].item()
        
        class_name = self.class_mapping.get(class_id, f"Class_{class_id}")
        
        return class_id, confidence, class_name
    
    def predict_batch(self, image_paths: List[str], confidence_threshold: float = 0.0) -> List[Dict]:
        """
        Predict from multiple images.
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence to report
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            class_id, confidence, class_name = self.predict_image(image_path)
            
            if confidence is not None and confidence >= confidence_threshold:
                results.append({
                    'image': str(image_path),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(confidence)
                })
        
        return results
    
    def close(self):
        """Release resources."""
        self.extractor.close()


def predict_single_image(image_path: str, model_path: str, device: str = None) -> Tuple[str, float]:
    """
    Simple function for single image prediction.
    
    Args:
        image_path: Path to image
        model_path: Path to model checkpoint
        device: Device to use
    
    Returns:
        Tuple of (class_name, confidence)
    """
    predictor = SignLanguagePredictor(model_path, device)
    class_id, confidence, class_name = predictor.predict_image(image_path)
    predictor.close()
    
    return class_name, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict sign language from images")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence to report prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (JSON)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SignLanguagePredictor(args.model, device=args.device)
    
    # Get image paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        image_paths = [str(p) for p in image_paths]
    else:
        logger.error(f"Invalid input path: {args.input}")
        return
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    # Predict
    results = predictor.predict_batch(image_paths, args.confidence_threshold)
    
    # Print results
    logger.info("="*60)
    logger.info("PREDICTIONS")
    logger.info("="*60)
    
    for result in results:
        logger.info(f"Image: {result['image']}")
        logger.info(f"  Prediction: {result['class_name']}")
        logger.info(f"  Confidence: {result['confidence']*100:.2f}%")
        logger.info()
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    predictor.close()


if __name__ == "__main__":
    main()
