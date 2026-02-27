"""
Smart City Surveillance — Server Entry Point

Development:
    python -m server.main

Production:
    gunicorn "server.main:app" -w 1 --threads 8 -b 0.0.0.0:8000

On startup the server will automatically connect to all cameras defined
in PI_CAMERAS inside server/api.py — no manual curl registration needed.
"""

from __future__ import annotations

import logging
import os

# Load .env file if present (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# api.py auto-registers all PI_CAMERAS when imported
from .api import app, manager, PI_CAMERAS
from flask import send_from_directory

# ─── Serve dashboard at / ────────────────────────────────────────────────────

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

@app.route("/")
def dashboard():
    return send_from_directory(os.path.abspath(DASHBOARD_DIR), "index.html")


# ─── Dev runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("SERVER_PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    log.info("=" * 55)
    log.info("  Smart City Surveillance Server")
    log.info("  Dashboard -> http://localhost:%d/", port)
    log.info("  Health    -> http://localhost:%d/health", port)
    log.info("  Cameras   -> %d auto-registered", len(PI_CAMERAS))
    for cam in PI_CAMERAS:
        log.info("    [%s] %s", cam["camera_id"], cam["stream_url"])
    log.info("=" * 55)

    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)