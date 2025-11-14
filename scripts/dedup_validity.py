#!/usr/bin/env python3
"""
Lightweight batch job to flag near-duplicate events within the recent window.
"""

# --- Phase F Patch --- ensure this helper can run standalone against SQLite
import os
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# --- Phase F Patch --- default to ingest_api/store/events.db unless overridden
DEFAULT_DB = os.environ.get(
    "INGEST_DB",
    str(Path(__file__).resolve().parent.parent / "ingest_api" / "store" / "events.db"),
)
LOOKBACK_NS = int(timedelta(minutes=10).total_seconds() * 1_000_000_000)
MAX_DT_NS = int(timedelta(seconds=5).total_seconds() * 1_000_000_000)
MAX_DBearing = 20.0


def _load_recent_events(conn: sqlite3.Connection) -> Sequence[sqlite3.Row]:
    """Fetch recent events for duplicate analysis."""
    now_ns = time.time_ns()
    min_rx_ns = now_ns - LOOKBACK_NS
    query = """
    SELECT id, event_id, ts_ns, rx_ns, bearing_deg, sensor_node_id
    FROM events
    WHERE rx_ns >= ?
    ORDER BY rx_ns DESC
    """
    return conn.execute(query, (min_rx_ns,)).fetchall()


def _duplicate_pairs(rows: Sequence[sqlite3.Row]) -> Iterable[str]:
    """Yield ONLY the newer event_id when two events (same node) are near-duplicates.
    Policy:
      - same sensor_node_id
      - |Δt| ≤ MAX_DT_NS (based on ts_ns)
      - circular Δbearing ≤ MAX_DBearing
    """
    for i, row in enumerate(rows):
        bearing_a = row["bearing_deg"]
        if bearing_a is None:
            continue
        for j in range(i + 1, len(rows)):
            other = rows[j]
            if row["sensor_node_id"] != other["sensor_node_id"]:
                continue
            bearing_b = other["bearing_deg"]
            if bearing_b is None:
                continue
            if abs(row["ts_ns"] - other["ts_ns"]) > MAX_DT_NS:
                continue
            if ang_diff(bearing_a, bearing_b) > MAX_DBearing:
                continue
            # mark only the later detection as duplicate
            newer = row if int(row["ts_ns"]) >= int(other["ts_ns"]) else other
            yield newer["event_id"]


def ang_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _apply_updates(conn: sqlite3.Connection, dup_ids: Iterable[str]) -> None:
    """Mark duplicates with safe parameterized UPDATE calls (do NOT touch validity)."""
    update_sql = "UPDATE events SET duplicate_flag = 1 WHERE event_id = ?"
    for event_id in dup_ids:
        conn.execute(update_sql, (event_id,))


def deduplicate(db_path: str = DEFAULT_DB) -> None:
    """Coordinator for the batch job."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_recent_events(conn)
        duplicates: List[str] = []
        for dup_id in _duplicate_pairs(rows):
            duplicates.append(dup_id)
        if duplicates:
            # --- Phase F Patch --- keep updates idempotent inside a single transaction
            _apply_updates(conn, set(duplicates))
            conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    deduplicate()
