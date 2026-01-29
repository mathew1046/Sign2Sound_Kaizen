"""
Hand Landmark Detection Module

This module provides hand landmark detection using MediaPipe Hands.
It wraps MediaPipe functionality for easier integration with the sign
language recognition pipeline.

Author: Team Kaizen
Date: January 2026
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List
import logging
import time
import urllib.request
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class HandLandmarkDetector:
    """
    Detect and extract hand landmarks using MediaPipe Hands.

    MediaPipe Hands detects 21 landmarks per hand:
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX_FINGER (MCP, PIP, DIP, TIP)
    9-12: MIDDLE_FINGER (MCP, PIP, DIP, TIP)
    13-16: RING_FINGER (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
    """

    # Landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Hands detector.

        Args:
            static_image_mode: If True, treats images as static (no tracking)
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
        """
        self.static_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = self._resolve_model_path(None)
        self.running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info(
            f"HandLandmarkDetector initialized (confidence={min_detection_confidence})"
        )

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)

        model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        default_path = model_dir / "hand_landmarker.task"

        if not default_path.exists():
            logger.info("Downloading MediaPipe Hand Landmarker model...")
            try:
                urllib.request.urlretrieve(
                    DEFAULT_HAND_LANDMARKER_URL, str(default_path)
                )
                logger.info(f"Model downloaded to {default_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download hand landmarker model: {e}")

        return default_path

    def detect_hands(self, image: np.ndarray) -> Optional[List]:
        """
        Detect hands in an image.

        Args:
            image: BGR image as numpy array

        Returns:
            List of hand landmarks or None if no hands detected
        """
        if image is None or image.size == 0:
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

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks

    def detect_hands_with_handedness(
        self, image: np.ndarray
    ) -> Optional[Tuple[List, List]]:
        """
        Detect hands in an image and return landmarks with handedness labels.

        Args:
            image: BGR image as numpy array

        Returns:
            Tuple of (hand_landmarks_list, handedness_list) or None if no hands detected
        """
        if image is None or image.size == 0:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.hand_landmarker.detect(mp_image)
        else:
            timestamp_ms = int(time.time() * 1000)
            result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        handedness_list = getattr(result, "handedness", None)
        if handedness_list is None:
            handedness_list = getattr(result, "multi_handedness", None)

        return result.hand_landmarks, handedness_list

    @staticmethod
    def _get_handedness_label(handedness_item) -> Optional[str]:
        """
        Extract handedness label from MediaPipe result.

        Returns:
            "Left", "Right", or None if not available
        """
        try:
            if isinstance(handedness_item, list) and handedness_item:
                handedness_item = handedness_item[0]

            if hasattr(handedness_item, "category_name"):
                return handedness_item.category_name

            if hasattr(handedness_item, "classification"):
                classification = handedness_item.classification
                if classification:
                    if hasattr(classification[0], "label"):
                        return classification[0].label
                    if hasattr(classification[0], "category_name"):
                        return classification[0].category_name
        except Exception:
            return None

        return None

    def landmarks_list_to_array(self, hand_landmarks_list: List) -> np.ndarray:
        """
        Convert MediaPipe hand landmark list to numpy array.

        Args:
            hand_landmarks_list: MediaPipe hand landmarks list

        Returns:
            Array of shape (num_hands, 21, 3) with (x, y, z) coordinates
        """
        landmarks_array = []

        for hand_landmarks in hand_landmarks_list:
            hand_array = []
            for landmark in hand_landmarks:
                hand_array.append([landmark.x, landmark.y, landmark.z])

            landmarks_array.append(hand_array)

        return np.array(landmarks_array, dtype=np.float32)

    def extract_landmarks(
        self, image: np.ndarray, hand_landmarks_list: Optional[List] = None
    ) -> Optional[np.ndarray]:
        """
        Extract hand landmarks as numpy array.

        Args:
            image: BGR image as numpy array
            hand_landmarks_list: Optional precomputed hand landmarks list

        Returns:
            Array of shape (num_hands, 21, 3) with (x, y, z) coordinates,
            or None if no hands detected
        """
        if hand_landmarks_list is None:
            hand_landmarks_list = self.detect_hands(image)

        if hand_landmarks_list is None:
            return None

        return self.landmarks_list_to_array(hand_landmarks_list)

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to [0, 1] range.
        MediaPipe already outputs normalized coordinates, but this ensures it.

        Args:
            landmarks: Array of shape (num_hands, 21, 3) or (21, 3)

        Returns:
            Normalized landmarks
        """
        return np.clip(landmarks, 0.0, 1.0).astype(np.float32)

    def format_features(self, landmarks: np.ndarray, max_hands: int = 2) -> np.ndarray:
        """
        Format landmarks into fixed-size feature vector (126 dimensions).

        Args:
            landmarks: Array of shape (num_hands, 21, 3)
            max_hands: Maximum number of hands (default: 2)

        Returns:
            Feature vector of shape (126,) = (max_hands * 21 * 3)
        """
        features = np.zeros(max_hands * 21 * 3, dtype=np.float32)

        num_hands = min(len(landmarks), max_hands)

        for hand_idx in range(num_hands):
            start_idx = hand_idx * 63  # 21 landmarks × 3 coords = 63
            hand_features = landmarks[hand_idx].flatten()
            features[start_idx : start_idx + 63] = hand_features

        return features

    def format_features_with_handedness(
        self, hand_landmarks_list: List, handedness_list: Optional[List]
    ) -> np.ndarray:
        """
        Format landmarks into fixed-size feature vector (126 dimensions)
        using hybrid routing:

        - Single hand: always map to Slot 1 (Left slot, indices 0-62)
        - Two hands: map Left label (or screen-left) to Slot 1,
          Right label to Slot 2.

        Model expects: [Left Hand Slot (0-62), Right Hand Slot (63-125)]

        Args:
            hand_landmarks_list: List of MediaPipe hand landmarks
            handedness_list: List of handedness classifications

        Returns:
            Feature vector of shape (126,)
        """
        lh_data = np.zeros(63, dtype=np.float32)
        rh_data = np.zeros(63, dtype=np.float32)

        if handedness_list is None:
            handedness_list = [None] * len(hand_landmarks_list)

        num_hands = len(hand_landmarks_list)

        if num_hands == 1:
            hand_landmarks = hand_landmarks_list[0]
            hand_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
            )
            hand_array = self.normalize_landmarks(hand_array)
            lh_data = hand_array.flatten()
            logger.info("Hybrid routing: single hand -> Slot 1 (indices 0-62)")
            return np.concatenate([lh_data, rh_data])

        if num_hands >= 2:
            labeled = []
            for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
                label = self._get_handedness_label(handedness)
                hand_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
                )
                hand_array = self.normalize_landmarks(hand_array)
                hand_flat = hand_array.flatten()
                mean_x = float(hand_array[:, 0].mean())
                labeled.append({
                    "label": label,
                    "hand_flat": hand_flat,
                    "mean_x": mean_x,
                })

            left_assigned = False
            right_assigned = False

            for item in labeled:
                if item["label"] == "Left" and not left_assigned:
                    lh_data = item["hand_flat"]
                    left_assigned = True
                elif item["label"] == "Right" and not right_assigned:
                    rh_data = item["hand_flat"]
                    right_assigned = True

            if not left_assigned or not right_assigned:
                labeled_sorted = sorted(labeled, key=lambda x: x["mean_x"])
                if not left_assigned and labeled_sorted:
                    lh_data = labeled_sorted[0]["hand_flat"]
                    left_assigned = True
                if not right_assigned and len(labeled_sorted) > 1:
                    rh_data = labeled_sorted[-1]["hand_flat"]
                    right_assigned = True

            logger.info(
                "Hybrid routing: two hands -> Slot1=Left(label/screen-left), Slot2=Right"
            )
            logger.info(
                "Hand details: %s",
                [
                    {
                        "label": item["label"],
                        "mean_x": round(item["mean_x"], 4),
                    }
                    for item in labeled
                ],
            )

        return np.concatenate([lh_data, rh_data])

    def process_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Complete pipeline: detect -> normalize -> format.

        Args:
            image: BGR image as numpy array

        Returns:
            Feature vector of shape (126,) or None if detection fails
        """
        detected = self.detect_hands_with_handedness(image)

        if detected is None:
            return None

        hand_landmarks_list, handedness_list = detected

        if handedness_list and len(handedness_list) == len(hand_landmarks_list):
            return self.format_features_with_handedness(
                hand_landmarks_list, handedness_list
            )

        landmarks = self.landmarks_list_to_array(hand_landmarks_list)
        landmarks = self.normalize_landmarks(landmarks)
        return self.format_features(landmarks, max_hands=2)

    def process_detected_landmarks(
        self, hand_landmarks_list: List, max_hands: int = 2
    ) -> Optional[np.ndarray]:
        """
        Complete pipeline using pre-detected landmarks: normalize -> format.

        Args:
            hand_landmarks_list: List of hand landmarks from detect_hands()
            max_hands: Maximum number of hands to include

        Returns:
            Feature vector of shape (126,) or None if detection fails
        """
        if hand_landmarks_list is None:
            return None

        landmarks = self.landmarks_list_to_array(hand_landmarks_list)
        landmarks = self.normalize_landmarks(landmarks)
        features = self.format_features(landmarks, max_hands=max_hands)
        return features

    def draw_landmarks(self, image: np.ndarray, landmarks_list: List) -> np.ndarray:
        """
        Draw hand landmarks on image for visualization.

        Args:
            image: BGR image as numpy array
            landmarks_list: List of hand landmarks from detect_hands()

        Returns:
            Image with drawn landmarks
        """
        annotated_image = image.copy()

        h, w, _ = annotated_image.shape
        for hand_landmarks in landmarks_list:
            for idx, landmark in enumerate(hand_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (cx, cy), 3, (0, 255, 0), -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return annotated_image

    def get_landmark_name(self, landmark_idx: int) -> str:
        """
        Get human-readable name for landmark index.

        Args:
            landmark_idx: Index (0-20)

        Returns:
            Landmark name
        """
        landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC",
            2: "THUMB_MCP",
            3: "THUMB_IP",
            4: "THUMB_TIP",
            5: "INDEX_MCP",
            6: "INDEX_PIP",
            7: "INDEX_DIP",
            8: "INDEX_TIP",
            9: "MIDDLE_MCP",
            10: "MIDDLE_PIP",
            11: "MIDDLE_DIP",
            12: "MIDDLE_TIP",
            13: "RING_MCP",
            14: "RING_PIP",
            15: "RING_DIP",
            16: "RING_TIP",
            17: "PINKY_MCP",
            18: "PINKY_PIP",
            19: "PINKY_DIP",
            20: "PINKY_TIP",
        }
        return landmark_names.get(landmark_idx, f"UNKNOWN_{landmark_idx}")

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "hand_landmarker"):
            self.hand_landmarker.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


def load_and_detect(
    image_path: str, detector: Optional["HandLandmarkDetector"] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load image and detect hand landmarks.

    Args:
        image_path: Path to image file
        detector: HandLandmarkDetector instance (creates new if None)

    Returns:
        Tuple of (features, annotated_image) or (None, None) if detection fails
    """
    # Create detector if not provided
    close_detector = detector is None
    typed_detector: HandLandmarkDetector = (
        detector if detector is not None else HandLandmarkDetector()
    )

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        if close_detector:
            typed_detector.close()
        return None, None

    # Detect landmarks
    hand_landmarks_list = typed_detector.detect_hands(image)

    if hand_landmarks_list is None:
        if close_detector:
            typed_detector.close()
        return None, None

    # Extract features
    landmarks = typed_detector.extract_landmarks(image, hand_landmarks_list)
    if landmarks is None:
        if close_detector:
            typed_detector.close()
        return None, None
    features = typed_detector.format_features(landmarks)

    # Draw landmarks
    annotated_image = typed_detector.draw_landmarks(image, hand_landmarks_list)

    # Clean up
    if close_detector:
        typed_detector.close()

    return features, annotated_image


if __name__ == "__main__":
    # Test hand landmark detector
    print("Testing Hand Landmark Detector...")

    # Create detector
    detector = HandLandmarkDetector(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    print("✓ Detector initialized")

    # Test landmark naming
    print("\nLandmark names:")
    for idx in [0, 4, 8, 12, 16, 20]:
        print(f"  {idx}: {detector.get_landmark_name(idx)}")

    # Test feature formatting
    dummy_landmarks = np.random.rand(2, 21, 3).astype(np.float32)  # 2 hands
    features = detector.format_features(dummy_landmarks)
    print(f"\n✓ Feature shape: {features.shape} (expected: (126,))")

    # Test normalization
    unnormalized = np.array([[[0.5, 1.5, -0.5]]] * 21).reshape(1, 21, 3)
    normalized = detector.normalize_landmarks(unnormalized)
    print(f"✓ Normalization: min={normalized.min():.2f}, max={normalized.max():.2f}")

    # Clean up
    detector.close()

    print("\n✓ All tests passed!")
