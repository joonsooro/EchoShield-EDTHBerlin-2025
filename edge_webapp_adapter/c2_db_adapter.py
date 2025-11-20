# edge_webapp_adapter/c2_db_adapter.py

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

from dsp_engine.node_output import NodeDetection
from dsp_engine.bearing_collect import BearingSet, collect_bearings_from_all_nodes


# --- Config -----------------------------------------------------------------

# Adjust if your DB file lives somewhere else
DB_PATH = Path(__file__).resolve().parent.parent/"ingest_api/store/events.db"

# Adjust if your table name is different (e.g. "events")
TABLE_NAME = "events"

EARTH_RADIUS_M = 6_371_000.0


# --- Small helpers ----------------------------------------------------------

def _latlon_to_local_xy(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:
    """
    Convert (lat, lon) in degrees into a local (x, y) tangent-plane coordinate
    in meters, using a simple equirectangular approximation.

    This is good enough for small areas (< ~5 km).
    """
    lat_r = math.radians(lat)
    lat0_r = math.radians(lat0)
    dlat = lat_r - lat0_r
    dlon = math.radians(lon - lon0)

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_r)  # east
    y = EARTH_RADIUS_M * dlat                     # north
    return x, y


# --- Core DB → NodeDetection conversion -------------------------------------

def fetch_recent_rows(
    lookback_sec: float = 10.0,
    min_confidence: float = 0.6,
    db_path: Path = DB_PATH,
) -> Sequence[sqlite3.Row]:
    """
    Fetch recent 'drone' events from edge_events.db.

    Assumptions:
    - Table has at least columns:
        node_id, ts_ns, azimuth_deg, confidence, event, lat, lon
    - ts_ns is UNIX time in nanoseconds (same epoch across nodes)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        now_ns = int(time.time() * 1e9)
        cutoff_ns = now_ns - int(lookback_sec * 1e9)

        sql = f"""
        SELECT
            sensor_node_id,
            ts_ns,
            bearing_deg,
            lat,
            lon
        FROM {TABLE_NAME}
        WHERE ts_ns >= 0
          AND bearing_deg IS NOT NULL
        ORDER BY ts_ns ASC
        """
        cur = conn.execute(sql, (cutoff_ns, min_confidence))
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def rows_to_detections_by_node(
    rows: Sequence[sqlite3.Row],
) -> Dict[str, List[NodeDetection]]:
    """
    Convert DB rows into NodeDetection objects grouped by node_id.

    Returns
    -------
    detections_by_node : dict[node_id -> List[NodeDetection]]
    """
    if not rows:
        return {}

    # Use the first valid lat/lon as local origin
    lat0 = rows[0]["lat"]
    lon0 = rows[0]["lon"]

    detections_by_node: Dict[str, List[NodeDetection]] = {}

    for r in rows:
        node_id = str(r["node_id"])
        ts_ns = int(r["ts_ns"])
        bearing_deg = float(r["bearing_deg"])
        confidence = float(r["confidence"])
        lat = float(r["lat"])
        lon = float(r["lon"])

        # Global timestamp in seconds
        t_global = ts_ns / 1e9

        # Project node position into local XY frame
        x_node, y_node = _latlon_to_local_xy(lat, lon, lat0, lon0)

        det = NodeDetection(
            node_id=node_id,
            t_global=t_global,
            x=x_node,
            y=y_node,
            bearing_local_deg=bearing_deg,   # already global in your JS, but keep API
            bearing_global_deg=bearing_deg,
            confidence=confidence,
            detected=True,
            snr_db=0.0,                      # DB에 없으면 0으로
        )

        detections_by_node.setdefault(node_id, []).append(det)

    return detections_by_node


# --- Convenience: DB → BearingSet ------------------------------------------

def load_bearing_sets_from_db(
    lookback_sec: float = 10.0,
    min_confidence: float = 0.6,
    fusion_dt: float = 0.2,
    min_nodes: int = 2,
    db_path: Path = DB_PATH,
) -> List[BearingSet]:
    """
    End-to-end helper:

    1. Read recent rows from edge_events.db
    2. Convert them to NodeDetection lists per node
    3. Run collect_bearings_from_all_nodes(..) to get BearingSet list
    """
    rows = fetch_recent_rows(
        lookback_sec=lookback_sec,
        min_confidence=min_confidence,
        db_path=db_path,
    )
    if not rows:
        return []

    detections_by_node = rows_to_detections_by_node(rows)
    if not detections_by_node:
        return []

    all_node_lists: List[List[NodeDetection]] = list(detections_by_node.values())

    bearing_sets = collect_bearings_from_all_nodes(
        all_node_detections=all_node_lists,
        fusion_dt=fusion_dt,
        min_nodes=min_nodes,
    )

    return bearing_sets
