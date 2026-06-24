"""OpenCV-DNN wrapper for cansik/yolo-hand-detection (bbox only)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class CansikHandDetector:
    """BBox-only hand detector from https://github.com/cansik/yolo-hand-detection."""

    def __init__(
        self,
        config: Path | str,
        weights: Path | str,
        size: int = 256,
        confidence: float = 0.2,
        threshold: float = 0.3,
    ):
        self.size = size
        self.confidence = confidence
        self.threshold = threshold
        self.net = cv2.dnn.readNetFromDarknet(str(config), str(weights))
        ln = self.net.getLayerNames()
        self.output_names = [ln[int(i) - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image: np.ndarray) -> tuple[list[tuple[float, int, int, int, int]], float]:
        """Return ``[(conf, x, y, w, h), ...]`` in pixel coords and inference seconds."""
        ih, iw = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        t0 = time.time()
        layer_outputs = self.net.forward(self.output_names)
        dt = time.time() - t0

        boxes: list[list[int]] = []
        confidences: list[float] = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                conf = float(scores[int(np.argmax(scores))])
                if conf <= self.confidence:
                    continue
                box = detection[0:4] * np.array([iw, ih, iw, ih])
                center_x, center_y, width, height = box.astype("int")
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(conf)

        results: list[tuple[float, int, int, int, int]] = []
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x, y, w, h = boxes[i]
                results.append((confidences[i], x, y, w, h))
        results.sort(key=lambda r: r[0], reverse=True)
        return results, dt


def expand_crop(
    x: int, y: int, w: int, h: int,
    frame_w: int, frame_h: int,
    pad_frac: float = 0.15,
) -> tuple[int, int, int, int]:
    pad_w = int(w * pad_frac)
    pad_h = int(h * pad_frac)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(frame_w, x + w + pad_w)
    y1 = min(frame_h, y + h + pad_h)
    return x0, y0, x1 - x0, y1 - y0
