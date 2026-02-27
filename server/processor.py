"""
Stream Processor
────────────────
Connects to a Pi MJPEG stream, pulls frames in a background thread,
runs all registered detectors, logs events, and exposes the latest
annotated frame as a JPEG byte-string for the Flask API to serve.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detectors.base import BaseDetector, Detection
from .db.mongo import log_detection, log_parking_event
from .utils.snapshot import save_snapshot

log = logging.getLogger(__name__)

# Maximum time (s) to wait for a fresh frame before retrying the stream.
READ_TIMEOUT  = 5
RETRY_DELAY   = 3
FRAME_RESIZE  = (640, 480)

# Run detectors only every N frames to keep stream fluid.
DETECTION_INTERVAL = 15   # <-- increased from 5


@dataclass
class CameraStats:
    camera_id:    str
    stream_url:   str
    frames_read:  int = 0
    detections:   int = 0
    errors:       int = 0
    fps:          float = 0.0
    connected:    bool = False
    last_frame_ts: float = field(default_factory=time.monotonic)


class StreamProcessor:
    """
    One instance per camera feed.

    Usage
    ─────
        proc = StreamProcessor(
            camera_id  = "cam-01",
            stream_url = "http://192.168.1.42:5000/video_feed",
            detectors  = [TrashDetector(), IllegalParkingDetector(zones=[...])],
        )
        proc.start()
    """

    def __init__(
        self,
        camera_id:  str,
        stream_url: str,
        detectors:  Optional[List[BaseDetector]] = None,
        save_snapshots: bool = True,
    ):
        self.camera_id      = camera_id
        self.stream_url     = stream_url
        self.detectors      = detectors or []
        self.save_snapshots = save_snapshots

        self.stats     = CameraStats(camera_id=camera_id, stream_url=stream_url)
        self._lock     = threading.Lock()
        self._latest   : Optional[bytes] = None   # JPEG bytes of last annotated frame
        self._raw_frame: Optional[np.ndarray] = None
        self._running  = False
        self._thread   : Optional[threading.Thread] = None

        # Store last detection results for drawing on skipped frames
        self._last_detections: List[Detection] = []

        # Cooldown for duplicate events
        self._last_event_time = {}          # key: (label, cx, cy) -> timestamp
        self._cooldown_seconds = 5          # seconds to wait before logging same object again

    # ─── Public API ───────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("[%s] Processor started → %s", self.camera_id, self.stream_url)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def get_latest_frame(self) -> Optional[bytes]:
        """Return the latest JPEG-encoded annotated frame (thread-safe)."""
        with self._lock:
            return self._latest

    def get_stats(self) -> dict:
        s = self.stats
        return {
            "camera_id":   s.camera_id,
            "stream_url":  s.stream_url,
            "frames_read": s.frames_read,
            "detections":  s.detections,
            "errors":      s.errors,
            "fps":         round(s.fps, 2),
            "connected":   s.connected,
        }

    # ─── Main loop ───────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            cap = self._open_stream()
            if cap is None:
                time.sleep(RETRY_DELAY)
                continue

            self.stats.connected = True
            fps_timer  = time.monotonic()
            fps_frames = 0
            frame_count = 0   # counter for detection interval

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("[%s] Frame read failed — reconnecting", self.camera_id)
                    self.stats.errors += 1
                    break

                frame = cv2.resize(frame, FRAME_RESIZE)
                self.stats.frames_read += 1
                fps_frames += 1
                frame_count += 1

                # FPS estimate every 30 frames
                if fps_frames >= 30:
                    elapsed = time.monotonic() - fps_timer
                    self.stats.fps = fps_frames / elapsed if elapsed else 0
                    fps_frames = 0
                    fps_timer  = time.monotonic()

                # Run detectors only every DETECTION_INTERVAL frames
                if frame_count % DETECTION_INTERVAL == 0:
                    # Full detection run
                    detections = self._run_detectors(frame)
                    self._last_detections = detections   # store for later
                else:
                    detections = self._last_detections   # reuse previous results

                # Annotate frame using the available detections
                annotated = self._draw_detections(frame, detections)
                self.stats.detections += len(detections)

                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self._lock:
                        self._latest = buf.tobytes()

            cap.release()
            self.stats.connected = False
            log.warning("[%s] Reconnecting in %ds…", self.camera_id, RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        log.info("[%s] Connecting to stream…", self.camera_id)
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.error("[%s] Cannot open stream %s", self.camera_id, self.stream_url)
            self.stats.errors += 1
            return None
        log.info("[%s] Stream opened.", self.camera_id)
        return cap

    def _run_detectors(self, frame: np.ndarray) -> List[Detection]:
        """Run all detectors on a fresh frame and persist events."""
        all_events: List[Detection] = []

        for detector in self.detectors:
            try:
                events = detector.detect(frame, camera_id=self.camera_id)
            except Exception as exc:
                log.exception("[%s] Detector %s raised: %s", self.camera_id, detector.name, exc)
                continue

            for det in events:
                all_events.append(det)
                self._persist(frame, det)   # <-- now includes cooldown

        return all_events

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes for the given detections onto a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.label == "illegal_parking" else (0, 200, 50)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{det.label} {det.confidence:.0%}"
            cv2.putText(annotated, text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated

    def _persist(self, raw_frame: np.ndarray, det: Detection):
        """Save snapshot and log event, with cooldown to avoid duplicates."""
        # Cooldown check: skip if same object was logged recently
        now = time.time()
        cx = (det.bbox[0] + det.bbox[2]) // 2
        cy = (det.bbox[1] + det.bbox[3]) // 2
        key = (det.label, cx, cy)
        last = self._last_event_time.get(key, 0)
        if now - last < self._cooldown_seconds:
            return
        self._last_event_time[key] = now

        snapshot_path: Optional[str] = None
        if self.save_snapshots:
            try:
                snapshot_path = save_snapshot(
                    raw_frame, det.label, self.camera_id, det.bbox
                )
            except Exception:
                log.exception("Snapshot save failed")

        try:
            if det.label == "illegal_parking":
                log_parking_event(det, snapshot_path)
            else:
                log_detection(det, snapshot_path)
        except Exception:
            log.exception("DB write failed for detection %s", det.label)


# ─── Multi-camera manager ────────────────────────────────────────────────────

class ProcessorManager:
    """Manages a pool of StreamProcessor instances."""

    def __init__(self):
        self._processors: Dict[str, StreamProcessor] = {}

    def add(self, processor: StreamProcessor):
        self._processors[processor.camera_id] = processor
        processor.start()

    def remove(self, camera_id: str):
        proc = self._processors.pop(camera_id, None)
        if proc:
            proc.stop()

    def get(self, camera_id: str) -> Optional[StreamProcessor]:
        return self._processors.get(camera_id)

    def all_stats(self) -> List[dict]:
        return [p.get_stats() for p in self._processors.values()]

    def stop_all(self):
        for proc in self._processors.values():
            proc.stop()
        self._processors.clear()