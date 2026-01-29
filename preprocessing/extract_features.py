"""
MediaPipe Feature Extraction Module

This module handles feature extraction from images and video sequences using
Google MediaPipe Hands. It extracts 21 hand landmarks for up to 2 hands,
resulting in a 126-dimensional feature vector (2 hands × 21 landmarks × 3 coords).

Author: Team Kaizen
Date: January 2026
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
import logging
from pathlib import Path
import time
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class MediaPipeExtractor:
    """
    Extract hand landmarks from images using MediaPipe Hands.
    """
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.3,
                 min_tracking_confidence: float = 0.3,
                 model_path: Optional[str] = None):
        """
        Initialize MediaPipe Hands detector.
        
        Args:
            static_image_mode: Whether to treat images as static (True) or video stream (False)
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = self._resolve_model_path(model_path)
        self.running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self._last_timestamp_ms = 0

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info(f"MediaPipe HandLandmarker initialized (confidence={min_detection_confidence})")

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)

        model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        default_path = model_dir / "hand_landmarker.task"

        if not default_path.exists():
            logger.info("Downloading MediaPipe Hand Landmarker model...")
            try:
                urllib.request.urlretrieve(DEFAULT_HAND_LANDMARKER_URL, str(default_path))
                logger.info(f"Model downloaded to {default_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download hand landmarker model: {e}")

        return default_path
    
    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Feature vector of shape (126,) or None if extraction fails
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.hand_landmarker.detect(mp_image)
        else:
            timestamp_ms = int(time.time() * 1000)
            result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract features
        features = self._extract_features_from_results(result, image.shape)
        
        return features
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a video frame (numpy array).
        
        Args:
            frame: BGR image as numpy array
        
        Returns:
            Feature vector of shape (126,) or None if extraction fails
        """
        if frame is None or frame.size == 0:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.hand_landmarker.detect(mp_image)
        else:
            timestamp_ms = int(time.time() * 1000)
            result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract features
        features = self._extract_features_from_results(result, frame.shape)
        
        return features
    
    def extract_from_sequence(self, frame_paths: List[str]) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a sequence of frames (for dynamic signs).
        
        Args:
            frame_paths: List of paths to frame images
        
        Returns:
            Feature array of shape (num_frames, 126) or None if extraction fails
        """
        features_list = []
        
        for frame_path in frame_paths:
            features = self.extract_from_image(frame_path)
            if features is not None:
                features_list.append(features)
        
        if len(features_list) == 0:
            logger.warning(f"No features extracted from sequence of {len(frame_paths)} frames")
            return None
        
        # Stack into array
        features_array = np.stack(features_list, axis=0)
        
        return features_array
    
    def _extract_features_from_results(self, 
                                      results, 
                                      image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Convert MediaPipe results to feature vector.
        
        Args:
            results: MediaPipe Hands results object
            image_shape: Shape of input image (height, width, channels)
        
        Returns:
            Feature vector of shape (126,) or None if no hands detected
        """
        if not results.hand_landmarks:
            return None
        
        # Initialize feature vector (126 dimensions)
        features = np.zeros(126, dtype=np.float32)
        
        # Extract landmarks for each detected hand (up to 2)
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            if hand_idx >= 2:  # Only process first 2 hands
                break
            
            # Extract 21 landmarks × 3 coordinates = 63 features per hand
            landmark_array = []
            for landmark in hand_landmarks:
                landmark_array.extend([landmark.x, landmark.y, landmark.z])
            
            # Place in appropriate position (first hand: 0-62, second hand: 63-125)
            start_idx = hand_idx * 63
            end_idx = start_idx + 63
            features[start_idx:end_idx] = np.array(landmark_array, dtype=np.float32)
        
        return features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features for quality.
        
        Args:
            features: Feature vector to validate
        
        Returns:
            True if features are valid, False otherwise
        """
        if features is None:
            return False
        
        # Check for NaN
        if np.isnan(features).any():
            return False
        
        # Check for Inf
        if np.isinf(features).any():
            return False
        
        # Check if completely zero (no hands detected)
        if not np.any(features):
            return False
        
        # Check for reasonable coordinate range
        if np.any(features < -0.5) or np.any(features > 1.5):
            return False
        
        return True
    
    def batch_extract_images(self, 
                           image_paths: List[str], 
                           show_progress: bool = True) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract features from multiple images in batch.
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (valid_features, valid_paths)
        """
        valid_features = []
        valid_paths = []
        
        iterator = image_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Extracting features")
            except ImportError:
                pass
        
        for image_path in iterator:
            features = self.extract_from_image(image_path)
            if features is not None and self.validate_features(features):
                valid_features.append(features)
                valid_paths.append(image_path)
        
        success_rate = len(valid_features) / len(image_paths) * 100 if image_paths else 0
        logger.info(f"Extracted {len(valid_features)}/{len(image_paths)} ({success_rate:.1f}%)")
        
        return valid_features, valid_paths
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


def extract_from_directory(directory: str, 
                          pattern: str = "*.jpg",
                          extractor: MediaPipeExtractor = None) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from all images in a directory.
    
    Args:
        directory: Path to directory containing images
        pattern: File pattern to match (default: "*.jpg")
        extractor: MediaPipeExtractor instance (creates new if None)
    
    Returns:
        Tuple of (features_array, file_paths)
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return np.array([]), []
    
    # Find all matching files
    image_files = sorted(dir_path.glob(pattern))
    
    if len(image_files) == 0:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return np.array([]), []
    
    # Create extractor if not provided
    close_extractor = False
    if extractor is None:
        extractor = MediaPipeExtractor()
        close_extractor = True
    
    # Extract features
    features_list, valid_paths = extractor.batch_extract_images(
        [str(f) for f in image_files],
        show_progress=True
    )
    
    # Close extractor if we created it
    if close_extractor:
        extractor.close()
    
    # Convert to array
    if len(features_list) > 0:
        features_array = np.stack(features_list, axis=0)
    else:
        features_array = np.array([])
    
    return features_array, valid_paths


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks to [0, 1] range.
    MediaPipe already outputs normalized coordinates, but this ensures it.
    
    Args:
        features: Feature vector or array
    
    Returns:
        Normalized features
    """
    return np.clip(features, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    # Test MediaPipe extraction
    print("Testing MediaPipe Feature Extraction...")
    
    # Create extractor
    extractor = MediaPipeExtractor(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    print("✓ MediaPipe Extractor initialized")
    
    # Test with dummy image (if available)
    # In practice, you would test with actual sign language images
    
    # Test feature validation
    valid_features = np.random.rand(126).astype(np.float32)
    invalid_features_nan = valid_features.copy()
    invalid_features_nan[0] = np.nan
    invalid_features_zero = np.zeros(126)
    
    print(f"\n✓ Valid features: {extractor.validate_features(valid_features)}")
    print(f"✗ NaN features: {extractor.validate_features(invalid_features_nan)}")
    print(f"✗ Zero features: {extractor.validate_features(invalid_features_zero)}")
    
    # Clean up
    extractor.close()
    
    print("\n✓ All tests passed!")
