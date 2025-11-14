-- ingest_api/store/models.sql
-- SQLite schema for canonical EchoShield events.
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS events (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id          TEXT NOT NULL,
  sensor_type       TEXT NOT NULL,        -- 'acoustic' | 'vision' | 'hybrid'
  ts_ns             INTEGER NOT NULL,     -- detection timestamp
  rx_ns             INTEGER NOT NULL,     -- receipt timestamp (server)
  latency_ns        INTEGER NOT NULL,     -- rx_ns - ts_ns (>=0 if clocks sane)
  latency_status    TEXT NOT NULL,        -- 'normal' | 'delayed' | 'obsolete'
  lat               REAL,                 -- nullable until triangulated/vision
  lon               REAL,
  error_radius_m    REAL,
  bearing_deg       REAL,                 -- nullable
  bearing_conf      REAL,                 -- 0..1 scale
  n_objects         INTEGER,
  event_code        TEXT,
  sensor_node_id    TEXT,
  location_method   TEXT,                 -- 'bearing_average'|'acoustic_triangulation'|'sensor_fusion'|'LOC_BEARING_ONLY'
  packet_version    INTEGER,
  validity_status   TEXT DEFAULT 'unknown', -- 'valid' | 'invalid' | 'unknown' (server-side result)
  duplicate_flag    INTEGER DEFAULT 0,      -- 0/1 (server-side result)
  object_track_id   TEXT,                   -- FK to tracks.track_id (nullable)
  clock_skew_ns     INTEGER,                -- rx_ns - ts_ns (can be negative)
  bearing_std_deg   REAL,                   -- stddev inside bearing cluster (optional)
  raw_wire_json     TEXT NOT NULL,        -- store original wire JSON for audit
  created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_rx_ns    ON events(rx_ns);
CREATE INDEX IF NOT EXISTS idx_events_node_ts  ON events(sensor_node_id, ts_ns);
CREATE INDEX IF NOT EXISTS idx_events_ts          ON events(ts_ns);
CREATE INDEX IF NOT EXISTS idx_events_track       ON events(object_track_id);
CREATE INDEX IF NOT EXISTS idx_events_created_at  ON events(created_at);

-- Aggregated tracks (bearing-only MVP)
CREATE TABLE IF NOT EXISTS tracks (
  track_id             TEXT PRIMARY KEY,         -- uuid
  method               TEXT,                     -- 'bearing_only' (for now)
  first_ts_ns          INTEGER,
  last_ts_ns           INTEGER,
  aggregated_bearing_deg REAL,                   -- bearing-only aggregated angle
  aggregated_lat       REAL,                     -- reserved for future triangulation
  aggregated_lon       REAL,                     -- reserved for future triangulation
  aggregation_conf     REAL,                     -- 0..1
  status               TEXT DEFAULT 'active'     -- 'active' | 'expired'
);

CREATE TABLE IF NOT EXISTS track_contributors (
  track_id        TEXT,
  event_id        TEXT,
  sensor_node_id  TEXT,
  bearing_deg     REAL,
  ts_ns           INTEGER,
  PRIMARY KEY (track_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_tracks_last ON tracks(last_ts_ns);

-- NOTE: Added clock_skew_ns to persist skew; ensure INSERT includes canonical.clock_skew_ns when available.
