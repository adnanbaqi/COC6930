"""
Trash / Litter Detector (COCO-based proxy) with full object detection.

Detects all objects from the YOLO model.
For classes that are likely litter (see TRASH_PROXY_LABELS), the label
is set to "trash_proxy" and the original class is stored in meta.
For all other classes, the original COCO label is used.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseDetector, Detection

log = logging.getLogger(__name__)

# COCO classes that can realistically represent litter
TRASH_PROXY_LABELS = {
    "bottle",
    "cup",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "wine glass",
    "fork",
    "knife",
    "spoon",
}

# Confidence threshold (adjust as needed)
CONFIDENCE_THRESHOLD = 0.35


class TrashDetector(BaseDetector):
    name = "trash"

    def __init__(self, model_path: str = "yolov8n.pt"):
        log.info("Loading YOLO model: %s", model_path)

        try:
            self._model = YOLO(model_path)
        except Exception as e:
            log.exception("Failed to load YOLO model")
            raise

        if torch.cuda.is_available():
            self._model.to("cuda")
            self._device = 0
            log.info("TrashDetector using GPU: %s",
                     torch.cuda.get_device_name(0))
        else:
            self._device = "cpu"
            log.warning("TrashDetector using CPU")

    def detect(self, frame: np.ndarray, camera_id: str = "unknown") -> List[Detection]:
        if frame is None:
            return []

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        try:
            results = self._model.predict(
                source=frame,
                device=self._device,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
        except Exception as e:
            log.exception("YOLO inference failed")
            return []

        detections: List[Detection] = []

        if results.boxes is None:
            return []

        for box in results.boxes:
            class_id = int(box.cls[0])
            label = results.names[class_id]
            conf = float(box.conf[0])

            log.debug(
                "[%s] RAW DETECTION → %s (%.2f)",
                camera_id,
                label,
                conf,
            )

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Determine final label:
            # - If the class is a trash proxy, mark it as "trash_proxy"
            # - Otherwise keep the original COCO class name
            if label.lower() in TRASH_PROXY_LABELS:
                detection_label = "trash_proxy"
                meta = {
                    "original_label": label,
                    "detector": self.name,
                }
            else:
                detection_label = label
                meta = {
                    "detector": self.name,
                }

            detections.append(
                Detection(
                    label=detection_label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    camera_id=camera_id,
                    meta=meta,
                )
            )

        if detections:
            log.info("[%s] Detections: %d", camera_id, len(detections))

        return detections