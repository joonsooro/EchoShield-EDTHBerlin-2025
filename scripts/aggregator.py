#!/usr/bin/env python3
"""
Minimal bearing-only track aggregator for EchoShield MVP data.
"""

# --- Phase F Patch --- helper entry-point for clustering events into tracks
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

WINDOW_NS = 10 * 1_000_000_000  # 10 seconds
MIN_CONTRIBUTORS = 2

DEFAULT_DB = os.environ.get(
    "INGEST_DB",
    str(Path(__file__).resolve().parent.parent / "ingest_api" / "store" / "events.db"),
)


def _load_bearing_events(conn: sqlite3.Connection) -> Sequence[sqlite3.Row]:
    """Pull recent bearing-only events that have not yet been assigned a track."""
    now_ns = time.time_ns()
    min_ts_ns = now_ns - WINDOW_NS * 6  # look back roughly one minute
    query = """
    SELECT event_id, ts_ns, bearing_deg, sensor_node_id, object_track_id
    FROM events
    WHERE ts_ns >= ?
      AND bearing_deg IS NOT NULL
    ORDER BY ts_ns ASC
    """
    return conn.execute(query, (min_ts_ns,)).fetchall()


def _cluster(rows: Sequence[sqlite3.Row]) -> Iterable[List[sqlite3.Row]]:
    """Group events by ts_ns bucket (10 second window)."""
    buckets: Dict[int, List[sqlite3.Row]] = {}
    for row in rows:
        bucket_key = int(row["ts_ns"] // WINDOW_NS)
        buckets.setdefault(bucket_key, []).append(row)
    for events in buckets.values():
        # Ensure multi-node contribution before yielding
        distinct_nodes = {r["sensor_node_id"] for r in events}
        if len(distinct_nodes) >= MIN_CONTRIBUTORS:
            yield events


def _track_id(bucket_key: int) -> str:
    """Derive deterministic track ids per bucket to keep logic simple."""
    return f"bearing-{bucket_key}"


def _upsert_track(conn: sqlite3.Connection, track_id: str, events: Sequence[sqlite3.Row]) -> None:
    bearings = [row["bearing_deg"] for row in events if row["bearing_deg"] is not None]
    aggregated_bearing = sum(bearings) / len(bearings)
    first_ts = min(row["ts_ns"] for row in events)
    last_ts = max(row["ts_ns"] for row in events)
    aggregation_conf = min(1.0, len(events) / 5)  # arbitrary lightweight scaling
    conn.execute(
        """
        INSERT INTO tracks (track_id, method, first_ts_ns, last_ts_ns, aggregated_bearing_deg, aggregation_conf)
        VALUES (?, 'bearing_only', ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
          last_ts_ns = excluded.last_ts_ns,
          aggregated_bearing_deg = excluded.aggregated_bearing_deg,
          aggregation_conf = excluded.aggregation_conf
        """,
        (track_id, first_ts, last_ts, aggregated_bearing, aggregation_conf),
    )


def _update_contributors(conn: sqlite3.Connection, track_id: str, events: Sequence[sqlite3.Row]) -> None:
    contributor_sql = """
    INSERT OR IGNORE INTO track_contributors (track_id, event_id, sensor_node_id, bearing_deg, ts_ns)
    VALUES (?, ?, ?, ?, ?)
    """
    update_event_sql = """
    UPDATE events
    SET object_track_id = ?
    WHERE event_id = ?
    """
    for row in events:
        conn.execute(
            contributor_sql,
            (track_id, row["event_id"], row["sensor_node_id"], row["bearing_deg"], row["ts_ns"]),
        )
        conn.execute(update_event_sql, (track_id, row["event_id"]))


def aggregate_tracks(db_path: str = DEFAULT_DB) -> None:
    """Main entry point for CLI usage."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_bearing_events(conn)
        for events in _cluster(rows):
            bucket_key = int(events[0]["ts_ns"] // WINDOW_NS)
            track_id = _track_id(bucket_key)
            # --- Phase F Patch --- keep insert/update logic scoped to each cluster batch
            _upsert_track(conn, track_id, events)
            _update_contributors(conn, track_id, events)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    aggregate_tracks()
