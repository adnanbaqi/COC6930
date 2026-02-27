# Smart City Surveillance System

Real-time video analytics pipeline.  
**Raspberry Pi** streams live video → **Central Server** runs YOLO detectors → events logged to **MongoDB**.

```
┌───────────────────┐        MJPEG over HTTP         ┌──────────────────────────────┐
│  Raspberry Pi     │  ──────────────────────────▶  │  Central Server (laptop/VM)  │
│                   │                                │                              │
│  pi_node/stream.py│                                │  YOLOv8 detectors            │
│  Flask MJPEG feed │                                │  ├── TrashDetector           │
│  /video_feed      │                                │  └── IllegalParkingDetector  │
│  /health          │                                │                              │
└───────────────────┘                                │  MongoDB logging             │
                                                     │  REST API  :8000             │
                                                     │  Dashboard /                 │
                                                     └──────────────────────────────┘
```

---

## Quick Start

### 1 — Raspberry Pi

```bash
# Install dependencies
pip install -r requirements_pi.txt

# Start streaming (default 640×480, port 5000)
python pi_node/stream.py

# Optional flags
python pi_node/stream.py --width 1280 --height 720 --quality 80 --port 5000
```

The Pi exposes:
| Route | Description |
|---|---|
| `/video_feed` | MJPEG stream |
| `/health` | JSON status & frame counter |

---

### 2 — Central Server

```bash
# Install dependencies
pip install -r requirements_server.txt

# Copy and fill in environment config
cp config/.env.example .env

# Start MongoDB (Docker example)
docker run -d -p 27017:27017 --name mongo mongo:7

# Start the server
python -m server.main
# or production:
gunicorn "server.main:app" -w 1 --threads 8 -b 0.0.0.0:8000  ## Server Linux Only
```

Open **http://localhost:8000** to see the dashboard.

---

### 3 — Register a Camera

```bash
curl -X POST http://localhost:8000/api/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id":   "cam-01",
    "stream_url":  "http://10.226.121.78:5000/video_feed",
    "parking_zones": [
      [[100, 300], [540, 300], [540, 480], [100, 480]]
    ]
  }'
```

`parking_zones` is a list of polygon point-lists in **pixel coordinates** of the incoming 640×480 frame.  
Omit the key to use the default zone (bottom third of the frame).

---

## REST API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cameras` | List cameras + stats |
| POST | `/api/cameras` | Register camera |
| DELETE | `/api/cameras/<id>` | Remove camera |
| GET | `/api/cameras/<id>/feed` | MJPEG annotated stream |
| GET | `/api/cameras/<id>/snapshot` | Latest JPEG frame |
| GET | `/api/parking/events` | Parking log (query: `camera_id`, `resolved`, `limit`) |
| POST | `/api/parking/events/<id>/resolve` | Mark event resolved |
| GET | `/api/stats` | Detection count by label |
| GET | `/health` | Server health |

---

## No-Parking Zones

Zones are defined as pixel polygons on the **640×480** normalized frame.  
To design zones visually, run:

```python
import cv2, json

frame = cv2.imread("reference_frame.jpg")
pts   = []

def click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("zone", frame)

cv2.imshow("zone", frame)
cv2.setMouseCallback("zone", click)
cv2.waitKey(0)
print(json.dumps([pts]))
```

Paste the printed JSON into the `parking_zones` field when registering a camera.

---

## MongoDB Collections

### `detections`
All detection events (trash, vehicles, etc.).

```json
{
  "label":      "bottle",
  "confidence": 0.82,
  "bbox":       [100, 200, 180, 350],
  "timestamp":  "2025-06-01T12:34:56",
  "camera_id":  "cam-01",
  "snapshot":   "snapshots/cam-01_bottle_20250601_123456_abc123.jpg",
  "meta":       { "detector": "trash" }
}
```

### `parking_logs`
Illegal parking events with resolution workflow.

```json
{
  "label":       "illegal_parking",
  "confidence":  0.91,
  "bbox":        [210, 310, 430, 470],
  "timestamp":   "2025-06-01T12:40:00",
  "camera_id":   "cam-01",
  "snapshot":    "snapshots/cam-01_illegal_parking_...",
  "resolved":    false,
  "resolved_at": null,
  "officer":     null,
  "notes":       null,
  "meta": {
    "vehicle_label": "car",
    "dwell_seconds": 15.3,
    "track_id":      4
  }
}
```

---

## Improving Detection Accuracy

| Goal | Action |
|---|---|
| Better vehicle detection | Use `yolov8s.pt` or `yolov8m.pt` |
| Detect trash specifically | Fine-tune on a trash dataset (TACO, OpenImages "Trash") |
| Reduce false positives | Increase `CONFIDENCE_THRESHOLD` in each detector |
| Faster Pi streaming | Lower `--quality` or `--width`/`--height` |

---

## Project Structure

```
smart-city-surveillance/
├── pi_node/
│   └── stream.py               # Raspberry Pi MJPEG server
├── server/
│   ├── api.py                  # Flask REST API
│   ├── processor.py            # Stream pull + detector orchestration
│   ├── main.py                 # Entry point
│   ├── detectors/
│   │   ├── base.py             # Detection dataclass + BaseDetector ABC
│   │   ├── trash_detector.py   # Litter / waste detector
│   │   └── parking_detector.py # Illegal parking with zone + dwell logic
│   ├── db/
│   │   └── mongo.py            # MongoDB read/write helpers
│   └── utils/
│       └── snapshot.py         # Evidence image saver
├── dashboard/
│   └── index.html              # Live monitoring dashboard
├── config/
│   └── .env.example
├── requirements_pi.txt
└── requirements_server.txt
```