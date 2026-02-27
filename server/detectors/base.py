"""
Base detector interface.
All detectors must inherit from BaseDetector and implement `detect()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: List[int]               # [x1, y1, x2, y2]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    camera_id: str = "unknown"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "confidence": round(self.confidence, 4),
            "bbox":       self.bbox,
            "timestamp":  self.timestamp.isoformat(),
            "camera_id":  self.camera_id,
            "meta":       self.meta,
        }


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, camera_id: str = "unknown") -> List[Detection]:
        """Run inference on a single BGR frame. Return a list of Detection objects."""
        ...

    def __repr__(self) -> str:
        return f"<Detector: {self.name}>"