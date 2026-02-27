import time
import logging
import argparse
import platform
from flask import Flask, Response, jsonify
import cv2

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pi-stream")

# ─── Config ─────────────────────────────────────────────────────────────────
DEFAULT_WIDTH    = 640
DEFAULT_HEIGHT   = 480
DEFAULT_FPS      = 15
DEFAULT_QUALITY  = 70   # JPEG compression quality
DEFAULT_PORT     = 5000

app = Flask(__name__)

# Will be initialised in main()
camera = None
STREAM_META: dict = {}


# ─── Camera abstraction ──────────────────────────────────────────────────────

def init_camera(width: int, height: int):
    """
    Use PiCamera2 when running on a Pi, fall back to OpenCV for dev/testing.
    """
    if platform.machine().startswith("aarch") or platform.machine() == "armv7l":
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            cam.configure(cfg)
            cam.start()
            time.sleep(2)
            log.info("PiCamera2 initialised at %dx%d", width, height)
            return cam, "picamera2"
        except Exception as exc:
            log.warning("PiCamera2 unavailable (%s), falling back to OpenCV", exc)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
    if not cap.isOpened():
        raise RuntimeError("No camera device found.")
    log.info("OpenCV camera initialised at %dx%d", width, height)
    return cap, "opencv"


def capture_frame(cam, backend: str, quality: int):
    """Return a JPEG-encoded frame as bytes, or None on failure."""
    if backend == "picamera2":
        import numpy as np
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


# ─── Generator ───────────────────────────────────────────────────────────────

def generate_frames():
    global camera, STREAM_META
    backend = STREAM_META["backend"]
    quality = STREAM_META["quality"]
    frame_count = 0

    while True:
        data = capture_frame(camera, backend, quality)
        if data is None:
            log.warning("Dropped frame #%d", frame_count)
            time.sleep(0.05)
            continue

        frame_count += 1
        STREAM_META["frames_served"] = frame_count

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + data
            + b"\r\n"
        )


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", **STREAM_META})


@app.route("/")
def index():
    return (
        "<h2>Pi Surveillance Node</h2>"
        '<p><a href="/video_feed">Video Feed</a></p>'
        '<p><a href="/health">Health</a></p>'
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    global camera, STREAM_META

    parser = argparse.ArgumentParser(description="Pi Surveillance Stream")
    parser.add_argument("--width",   type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height",  type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--port",    type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    camera, backend = init_camera(args.width, args.height)

    STREAM_META = {
        "backend":       backend,
        "resolution":    f"{args.width}x{args.height}",
        "quality":       args.quality,
        "frames_served": 0,
        "node_id":       platform.node(),
    }

    log.info("Streaming on port %d  [backend=%s]", args.port, backend)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()