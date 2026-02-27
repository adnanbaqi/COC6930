"""
Central Server — Flask REST API

Cameras are auto-registered on startup using the hardcoded Pi stream URL.
Manual registration via POST /api/cameras is also supported for additional cameras.

Routes
------
GET    /api/cameras                      List all cameras + live stats
POST   /api/cameras                      Manually register an extra camera
DELETE /api/cameras/<id>                 Remove a camera
GET    /api/cameras/<id>/feed            MJPEG annotated live stream
GET    /api/cameras/<id>/snapshot        Latest JPEG frame
GET    /api/parking/events               Parking log (filter: camera_id, resolved, limit)
POST   /api/parking/events/<id>/resolve  Mark a parking event as resolved
GET    /api/stats                        Detection counts by label
GET    /health                           Health check
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from flask import Flask, Response, jsonify, request, abort

from .processor import ProcessorManager, StreamProcessor
from .detectors.trash_detector import TrashDetector
from .detectors.parking_detector import IllegalParkingDetector
from .db.mongo import get_parking_events, resolve_parking_event, get_detection_stats

log = logging.getLogger(__name__)

# ─── Hardcoded Pi camera config ───────────────────────────────────────────────

PI_CAMERAS = [
    {
        "camera_id":     "pi-cam-01",
        "stream_url":    "http://172.20.10.3:5000/video_feed",
        # No-parking zone: bottom third of 640x480 frame (pixel coordinates).
        # Edit these points to match your actual restricted area.
        "parking_zones": [
            [(0, 320), (640, 320), (640, 480), (0, 480)]
        ],
    },
        {
        "camera_id": "laptop-cam-01",
        "stream_url": "http://127.0.0.1:5123/video_feed",
        "parking_zones": [
            [(0, 320), (640, 320), (640, 480), (0, 480)]
        ],
    },
    # Add more Pi cameras here if needed:
    # {
    #     "camera_id":     "pi-cam-02",
    #     "stream_url":    "http://192.168.1.xx:5000/video_feed",
    #     "parking_zones": [[(100, 200), (540, 200), (540, 480), (100, 480)]],
    # },
]

# ─── App + manager ────────────────────────────────────────────────────────────

app     = Flask(__name__)
manager = ProcessorManager()


# ─── Detector factory ─────────────────────────────────────────────────────────

def _build_detectors(parking_zones: Optional[list] = None) -> list:
    """
    Build the detector stack for a camera.
    parking_zones: list of polygon point-lists in 640x480 pixel coords.
    """
    zones = parking_zones or [[(0, 320), (640, 320), (640, 480), (0, 480)]]
    return [
        TrashDetector(),
        IllegalParkingDetector(zones=zones),
    ]


def _register(camera_id: str, stream_url: str, parking_zones: Optional[list] = None):
    """Create a StreamProcessor and add it to the manager."""
    proc = StreamProcessor(
        camera_id=camera_id,
        stream_url=stream_url,
        detectors=_build_detectors(parking_zones),
    )
    manager.add(proc)
    log.info("Camera registered: %s -> %s", camera_id, stream_url)


# ─── Auto-register Pi cameras on import ──────────────────────────────────────

for _cam in PI_CAMERAS:
    _register(
        camera_id=    _cam["camera_id"],
        stream_url=   _cam["stream_url"],
        parking_zones=_cam.get("parking_zones"),
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify(manager.all_stats())


@app.route("/api/cameras", methods=["POST"])
def register_camera():
    """Manually register an additional camera at runtime."""
    body      = request.get_json(force=True)
    camera_id = body.get("camera_id")
    url       = body.get("stream_url")
    zones     = body.get("parking_zones")

    if not camera_id or not url:
        abort(400, "camera_id and stream_url are required")
    if manager.get(camera_id):
        abort(409, f"Camera '{camera_id}' is already registered")

    _register(camera_id, url, zones)
    return jsonify({"status": "ok", "camera_id": camera_id}), 201


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def unregister_camera(camera_id):
    if not manager.get(camera_id):
        abort(404, f"Camera '{camera_id}' not found")
    manager.remove(camera_id)
    return jsonify({"status": "removed", "camera_id": camera_id})


# ─── Live feed + snapshot ─────────────────────────────────────────────────────

@app.route("/api/cameras/<camera_id>/feed")
def camera_feed(camera_id):
    proc = manager.get(camera_id)
    if proc is None:
        abort(404, f"Camera '{camera_id}' not found")

    def generate():
        # Wait for the first frame (max 10 seconds)
        timeout = 10
        start = time.monotonic()
        frame = None
        while frame is None and (time.monotonic() - start) < timeout:
            frame = proc.get_latest_frame()
            if frame is None:
                time.sleep(0.1)   # don't burn CPU

        if frame is None:
            log.error("[%s] No frame available after %ds", camera_id, timeout)
            # Yield an error frame? For now, just stop the generator (client will retry)
            return

        # Yield the first frame, then continue as normal
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        while True:
            frame = proc.get_latest_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                # If we lose the stream, wait a bit before retrying
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/cameras/<camera_id>/snapshot")
def camera_snapshot(camera_id):
    proc = manager.get(camera_id)
    if proc is None:
        abort(404, f"Camera '{camera_id}' not found")
    frame = proc.get_latest_frame()
    if frame is None:
        abort(503, "No frame available yet — stream may still be connecting")
    return Response(frame, mimetype="image/jpeg")


# ─── Parking events ───────────────────────────────────────────────────────────

@app.route("/api/parking/events", methods=["GET"])
def list_parking():
    camera_id    = request.args.get("camera_id")
    resolved_str = request.args.get("resolved")
    limit        = int(request.args.get("limit", 100))

    resolved_bool: Optional[bool] = None
    if resolved_str is not None:
        resolved_bool = resolved_str.lower() == "true"

    events = get_parking_events(
        camera_id=camera_id,
        resolved=resolved_bool,
        limit=limit,
    )
    return jsonify(events)


@app.route("/api/parking/events/<event_id>/resolve", methods=["POST"])
def resolve_event(event_id):
    body    = request.get_json(force=True)
    officer = body.get("officer", "unknown")
    notes   = body.get("notes", "")

    ok = resolve_parking_event(event_id, officer=officer, notes=notes)
    if not ok:
        abort(404, "Event not found or already resolved")
    return jsonify({"status": "resolved", "event_id": event_id})


# ─── Stats + health ───────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    return jsonify(get_detection_stats())


@app.route("/health")
def health():
    cams = manager.all_stats()
    return jsonify({
        "status":        "ok",
        "cameras_total": len(cams),
        "cameras_live":  sum(1 for c in cams if c["connected"]),
    })


# ─── Cleanup on shutdown ──────────────────────────────────────────────────────

import atexit
atexit.register(manager.stop_all)