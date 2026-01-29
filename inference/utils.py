"""
Inference Utilities

Helper functions for inference and prediction postprocessing.

Author: Team Kaizen
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_class_mapping(csv_path: str = 'data/processed/class_mapping.csv') -> Dict[int, str]:
    """
    Load class mapping from CSV.
    
    Args:
        csv_path: Path to class_mapping.csv
    
    Returns:
        Dictionary mapping class_idx -> class_name
    """
    try:
        df = pd.read_csv(csv_path)
        class_mapping = dict(zip(df['class_idx'], df['class_name']))
        logger.info(f"Loaded {len(class_mapping)} classes from {csv_path}")
        return class_mapping
    except Exception as e:
        logger.error(f"Failed to load class mapping: {e}")
        return {}


def preprocess_image(image_path: str, extractor) -> np.ndarray:
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to image
        extractor: HandLandmarkDetector instance
    
    Returns:
        Processed feature vector
    """
    features = extractor.process_image(str(image_path))
    
    if features is None:
        logger.warning(f"Failed to extract features from {image_path}")
        return None
    
    # Pad to (60, 126)
    from features.feature_utils import pad_or_truncate
    features = pad_or_truncate(features, max_len=60)
    
    return features


def postprocess_prediction(logits: np.ndarray,
                          confidence_threshold: float = 0.0) -> Dict:
    """
    Postprocess model output.
    
    Args:
        logits: Model output logits
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Dictionary with class_id, confidence, class_name
    """
    # Apply softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # Get top predictions
    top_idx = np.argsort(probs[0])[::-1]
    
    predictions = []
    for idx in top_idx[:3]:  # Top 3
        confidence = float(probs[0, idx])
        
        if confidence >= confidence_threshold:
            predictions.append({
                'class_id': int(idx),
                'confidence': confidence
            })
    
    return predictions


def format_prediction_output(class_id: int,
                            confidence: float,
                            class_mapping: Dict[int, str]) -> str:
    """
    Format prediction for display.
    
    Args:
        class_id: Predicted class ID
        confidence: Prediction confidence
        class_mapping: Class name mapping
    
    Returns:
        Formatted string
    """
    class_name = class_mapping.get(class_id, f"Class_{class_id}")
    
    # Extract clean name (remove Malayalam_ or ISL_ prefix)
    clean_name = class_name.replace('Malayalam_', '').replace('ISL_', '')
    
    return f"{clean_name} ({confidence*100:.1f}%)"


def save_predictions(predictions: list, output_path: str):
    """
    Save predictions to JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")


def load_predictions(json_path: str) -> list:
    """
    Load predictions from JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        List of prediction dictionaries
    """
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    return predictions


if __name__ == "__main__":
    print("Inference utilities module")
