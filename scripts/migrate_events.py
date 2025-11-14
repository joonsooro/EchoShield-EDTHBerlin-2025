#!/usr/bin/env python3
import sqlite3, os

DB = "ingest_api/store/events.db"

cols_needed = {
    "validity_status": "TEXT DEFAULT 'unknown'",
    "duplicate_flag": "INTEGER DEFAULT 0",
    "object_track_id": "TEXT",
    "clock_skew_ns": "INTEGER",
    "bearing_std_deg": "REAL",
}

def have_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def ensure_columns(conn, table, spec):
    have = have_columns(conn, table)
    for col, ddl in spec.items():
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

def ensure_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
      track_id TEXT PRIMARY KEY,
      method TEXT,
      first_ts_ns INTEGER,
      last_ts_ns INTEGER,
      aggregated_bearing_deg REAL,
      aggregated_lat REAL,
      aggregated_lon REAL,
      aggregation_conf REAL,
      status TEXT DEFAULT 'active'
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS track_contributors (
      track_id TEXT,
      event_id TEXT,
      sensor_node_id TEXT,
      bearing_deg REAL,
      ts_ns INTEGER,
      PRIMARY KEY (track_id, event_id)
    )""")

def ensure_indexes(conn):
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_track ON events(object_track_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_last ON tracks(last_ts_ns)")

if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    ensure_columns(conn, "events", cols_needed)
    ensure_tables(conn)
    ensure_indexes(conn)
    conn.commit()
    conn.close()
    print("✅ Migration completed for", DB)
