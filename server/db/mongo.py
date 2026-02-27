"""
MongoDB persistence layer.

Collections
───────────
  detections   — every raw detection event
  parking_logs — enriched illegal-parking events with snapshot path
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from ..detectors.base import Detection

log = logging.getLogger(__name__)

# ─── Connection ──────────────────────────────────────────────────────────────

_client: Optional[MongoClient] = None
_db     = None

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "smart_city")


def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        _db     = _client[DB_NAME]
        _ensure_indexes()
        log.info("MongoDB connected → %s / %s", MONGO_URI, DB_NAME)
    return _db


def _ensure_indexes():
    db = _db
    db.detections.create_index([("timestamp", DESCENDING)])
    db.detections.create_index([("label", 1)])
    db.detections.create_index([("camera_id", 1)])
    db.parking_logs.create_index([("timestamp", DESCENDING)])
    db.parking_logs.create_index([("camera_id", 1)])
    db.parking_logs.create_index([("resolved", 1)])
    log.debug("Indexes ensured.")


# ─── Public helpers ──────────────────────────────────────────────────────────

def log_detection(detection: Detection, snapshot_path: Optional[str] = None):
    """Insert any detection event into the detections collection."""
    doc = detection.to_dict()
    if snapshot_path:
        doc["snapshot"] = snapshot_path
    get_db().detections.insert_one(doc)


def log_parking_event(detection: Detection, snapshot_path: Optional[str] = None):
    """
    Insert an illegal-parking event into the dedicated parking_logs collection.
    Includes a 'resolved' flag for later officer acknowledgement.
    """
    doc = {
        **detection.to_dict(),
        "snapshot":   snapshot_path,
        "resolved":   False,
        "resolved_at": None,
        "officer":    None,
        "notes":      None,
    }
    result = get_db().parking_logs.insert_one(doc)
    log.info("Parking event logged → _id=%s camera=%s", result.inserted_id, detection.camera_id)
    return str(result.inserted_id)


def resolve_parking_event(event_id: str, officer: str, notes: str = "") -> bool:
    """Mark a parking event as resolved."""
    from bson import ObjectId
    db     = get_db()
    result = db.parking_logs.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": {
            "resolved":    True,
            "resolved_at": datetime.utcnow().isoformat(),
            "officer":     officer,
            "notes":       notes,
        }},
    )
    return result.modified_count == 1


def get_parking_events(
    camera_id: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
) -> list:
    query: dict = {}
    if camera_id is not None:
        query["camera_id"] = camera_id
    if resolved is not None:
        query["resolved"] = resolved

    cursor = (
        get_db()
        .parking_logs
        .find(query, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


def get_detection_stats() -> dict:
    db = get_db()
    pipeline = [
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    labels = {r["_id"]: r["count"] for r in db.detections.aggregate(pipeline)}
    total  = db.detections.count_documents({})
    return {"total": total, "by_label": labels}