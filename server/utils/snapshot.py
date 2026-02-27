"""
Snapshot utility — saves annotated frames as evidence images.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(
    frame: np.ndarray,
    label: str,
    camera_id: str = "unknown",
    bbox: Optional[list] = None,
) -> str:
    """
    Save an annotated frame to disk.
    Returns the file path.
    """
    annotated = frame.copy()

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
        )

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid      = uuid.uuid4().hex[:6]
    filename = f"{camera_id}_{label}_{ts}_{uid}.jpg"
    path     = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path