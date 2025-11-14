# ui/app.py
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import math
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# --- Config ---
DEFAULT_DB = os.environ.get(
    "INGEST_DB",
    os.path.join(os.path.dirname(__file__), "..", "ingest_api", "store", "events.db"),
)
REFRESH_SECS = int(os.environ.get("UI_REFRESH_SECS", "5"))  # auto-refresh interval

st.set_page_config(page_title="EchoShield UI (MVP)", layout="wide")

# Geo helpers for bearing corridor ---
EARTH_R = 6371000.0  # meters

def _dest_point(lat, lon, bearing_deg, dist_m):
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = dist_m / EARTH_R
    lat2 = math.asin(math.sin(lat1)*math.cos(dr) + math.cos(lat1)*math.sin(dr)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def add_bearing_corridor(m, lat, lon, bearing_deg, half_width_deg=20.0, radius_m=500.0):
    """Draw a wedge (center ray ± half_width) from (lat,lon) on folium.Map."""
    if bearing_deg is None:
        return
    left_b  = (bearing_deg - half_width_deg) % 360
    right_b = (bearing_deg + half_width_deg) % 360
    # edge points
    lat_c, lon_c = _dest_point(lat, lon, bearing_deg, radius_m)
    lat_l, lon_l = _dest_point(lat, lon, left_b,   radius_m)
    lat_r, lon_r = _dest_point(lat, lon, right_b,  radius_m)
    # wedge polygon (center → left → right → back)
    folium.Polygon(
        locations=[(lat, lon), (lat_l, lon_l), (lat_c, lon_c), (lat_r, lon_r)],
        fill=True, fill_opacity=0.15, weight=1
    ).add_to(m)
    # center ray
    folium.PolyLine([(lat, lon), (lat_c, lon_c)], weight=2).add_to(m)

# --- Helpers ---
@st.cache_data(ttl=0)  # short cache to avoid hammering
def load_events(db_path: str, lookback_minutes: int, seed: int) -> pd.DataFrame:
    # seed is only used for cache key.
    if not os.path.exists(db_path):
        return pd.DataFrame(
            columns=[
                "id",
                "event_id",
                "sensor_type",
                "ts_ns",
                "rx_ns",
                "latency_ns",
                "latency_status",
                "lat",
                "lon",
                "error_radius_m",
                "bearing_deg",
                "bearing_conf",
                "n_objects",
                "event_code",
                "sensor_node_id",
                "location_method",
                "packet_version",
                "validity_status",
                "duplicate_flag",
                "object_track_id",
                "created_at",
            ]
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if lookback_minutes > 0:
            # ns based filter: Current time(ns) - Look back time(ns)
            now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            min_rx_ns = now_ns - lookback_minutes * 60 * 1_000_000_000
            query = """
            SELECT id, event_id, sensor_type, ts_ns, rx_ns, latency_ns, latency_status,
                   lat, lon, error_radius_m, bearing_deg, bearing_conf, n_objects,
                   event_code, sensor_node_id, location_method, packet_version,
                   validity_status, duplicate_flag, object_track_id, created_at
            FROM events
            WHERE rx_ns >= ?
            ORDER BY id DESC
            LIMIT 2000
            """
            df = pd.read_sql_query(query, conn, params=[min_rx_ns])
        else:
            query = """
            SELECT id, event_id, sensor_type, ts_ns, rx_ns, latency_ns, latency_status,
                   lat, lon, error_radius_m, bearing_deg, bearing_conf, n_objects,
                   event_code, sensor_node_id, location_method, packet_version,
                   validity_status, duplicate_flag, object_track_id, created_at
            FROM events
            ORDER BY id DESC
            LIMIT 2000
            """
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


# Fetch latest active tracks for tab rendering
@st.cache_data(ttl=0)
def load_tracks(db_path: str) -> pd.DataFrame:
    columns = [
        "track_id",
        "method",
        "first_ts_ns",
        "last_ts_ns",
        "aggregated_bearing_deg",
        "aggregated_lat",
        "aggregated_lon",
        "aggregation_conf",
        "status",
    ]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=columns)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
        SELECT track_id, method, first_ts_ns, last_ts_ns, aggregated_bearing_deg,
               aggregated_lat, aggregated_lon, aggregation_conf, status
        FROM tracks
        WHERE status = 'active'
        ORDER BY last_ts_ns DESC
        LIMIT 200
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# Capture contributor rows for additional context
@st.cache_data(ttl=0)
def load_track_contributors(db_path: str) -> pd.DataFrame:
    columns = ["track_id", "event_id", "sensor_node_id", "bearing_deg", "ts_ns"]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=columns)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
        SELECT track_id, event_id, sensor_node_id, bearing_deg, ts_ns
        FROM track_contributors
        ORDER BY ts_ns DESC
        LIMIT 500
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# --- Phase F Patch --- simple styling helper for new validity columns
def style_events_table(df: pd.DataFrame):
    def validity_style(val):
        if val == "invalid": return "background-color:#f8d7da"
        if val == "valid":   return "background-color:#d1e7dd"
        return ""

    def duplicate_style(val):
        if pd.isna(val): return ""
        return "background-color:#fff3cd" if int(val) else ""

    def track_style(val):
        return "background-color:#dbeafe" if pd.notna(val) else ""

    styler = df.style
    if "validity_status" in df.columns:
        styler = styler.map(lambda v: validity_style(v), subset=["validity_status"])
    if "duplicate_flag" in df.columns:
        styler = styler.map(lambda v: duplicate_style(v), subset=["duplicate_flag"])
    if "object_track_id" in df.columns:
        styler = styler.map(lambda v: track_style(v), subset=["object_track_id"])
    return styler


def ns_to_ms(ns: Optional[int]) -> Optional[float]:
    try:
        return round(ns / 1_000_000.0, 3) if ns is not None else None
    except Exception:
        return None


def kpis(df: pd.DataFrame) -> Tuple[int, int, float, float, float]:
    total = len(df)
    nodes = df["sensor_node_id"].nunique() if total else 0
    if total:
        lat_ms = df["latency_ns"].apply(ns_to_ms)
        p50 = float(lat_ms.quantile(0.5))
        p95 = float(lat_ms.quantile(0.95))
        p99 = float(lat_ms.quantile(0.99))
    else:
        p50 = p95 = p99 = 0.0
    return total, nodes, p50, p95, p99


# --- UI Patch: latency distribution helpers ---
@st.cache_data(ttl=2)
def latency_ms_series(df: pd.DataFrame) -> pd.Series:
    if df.empty or "latency_ns" not in df.columns:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(df["latency_ns"], errors="coerce").astype("float64") / 1_000_000.0
    return s.clip(lower=0)

def p95_by_node(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sensor_node_id" not in df.columns:
        return pd.DataFrame(columns=["sensor_node_id", "p95_ms"])
    tmp = df.copy()
    tmp["latency_ms"] = pd.to_numeric(tmp["latency_ns"], errors="coerce").astype("float64") / 1_000_000.0
    tmp["latency_ms"] = tmp["latency_ms"].clip(lower=0)
    out = tmp.groupby("sensor_node_id")["latency_ms"].quantile(0.95).reset_index(name="p95_ms")
    out["p95_ms"] = out["p95_ms"].round(1)
    return out.sort_values("p95_ms", ascending=False)

# --- UI Patch: node health helper ---
@st.cache_data(ttl=2)
def node_health(df: pd.DataFrame, lookback_min: int = 5) -> pd.DataFrame:
    if df.empty or "rx_ns" not in df.columns:
        return pd.DataFrame(columns=["sensor_node_id", "last_rx", "age_sec", "status"])
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    last_rx = df.groupby("sensor_node_id")["rx_ns"].max().reset_index()
    last_rx["age_sec"] = ((now_ns - pd.to_numeric(last_rx["rx_ns"], errors="coerce")) / 1e9).round(1)
    last_rx["last_rx"] = pd.to_datetime(last_rx["rx_ns"], unit="ns", utc=True).dt.tz_convert(None)
    last_rx["status"] = last_rx["age_sec"].apply(lambda s: "WARN" if s > lookback_min * 60 else "OK")
    return last_rx.sort_values("age_sec", ascending=False)


def render_map(df: pd.DataFrame, draw_corridor: bool = False, half_width_deg: float = 20.0, radius_m: float = 500.0):
    df_map = df.dropna(subset=["lat", "lon"]).copy()
    # Cast to float and drop meaningless (0,0) coords
    df_map["lat"] = df_map["lat"].astype(float)
    df_map["lon"] = df_map["lon"].astype(float)
    df_map = df_map[~((df_map["lat"] == 0.0) & (df_map["lon"] == 0.0))]
    if df_map.empty:
        st.info("No geolocated events in the selected window.")
        return
    c_lat = float(df_map["lat"].mean())
    c_lon = float(df_map["lon"].mean())

    # HTTPS tile + attribution
    m = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=10,
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors"
    )
    for _, row in df_map.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        popup = folium.Popup(
            f"""
            <b>event_id:</b> {row.get('event_id','')}<br/>
            <b>node:</b> {row.get('sensor_node_id','')}<br/>
            <b>latency(ms):</b> {ns_to_ms(row.get('latency_ns'))}<br/>
            <b>bearing:</b> {row['bearing_deg'] if pd.notna(row.get('bearing_deg')) else 'n/a'}<br/>
            <b>status:</b> {row.get('latency_status','')}
            """,
            max_width=300,
        )
        folium.CircleMarker(location=[lat, lon], radius=6).add_to(m)
        folium.Marker([lat, lon], popup=popup).add_to(m)

        # Optional bearing corridor wedge (uses existing add_bearing_corridor)
        if draw_corridor and pd.notna(row.get("bearing_deg")):
            try:
                add_bearing_corridor(
                    m,
                    lat,
                    lon,
                    float(row["bearing_deg"]),
                    half_width_deg=float(half_width_deg),
                    radius_m=float(radius_m),
                )
            except Exception:
                pass
    st_folium(m, height=500, use_container_width=True)


# --- Sidebar controls ---
st.sidebar.title("EchoShield — Controls")
db_path = st.sidebar.text_input("SQLite DB path", DEFAULT_DB)
lookback = st.sidebar.slider("Lookback window (minutes)", 0, 240, 30, step=5)
autorefresh = st.sidebar.checkbox(f"Auto-refresh every {REFRESH_SECS}s", value=True)

# --- Auto-refresh (version-agnostic) ---
if autorefresh:
    # Browser-level refresh every REFRESH_SECS seconds
    st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_SECS}'>", unsafe_allow_html=True)
    st.caption(f"Auto-refresh every {REFRESH_SECS}s")

# Use a changing 'seed' so @st.cache_data(ttl=0) invalidates each interval
refresh_seed = int(time.time() // REFRESH_SECS) if autorefresh else int(time.time())

# --- Data load ---
df = load_events(db_path, lookback, refresh_seed)

# --- KPI Row ---
st.title("EchoShield — MVP UI")
col1, col2, col3, col4, col5 = st.columns(5)
total, nodes, p50, p95, p99 = kpis(df)
col1.metric("Events", total)
col2.metric("Nodes", nodes)
col3.metric("Latency p50 (ms)", f"{p50:.1f}")
col4.metric("Latency p95 (ms)", f"{p95:.1f}")
col5.metric("Latency p99 (ms)", f"{p99:.1f}")

# --- Filters ---
c1, c2, c3 = st.columns(3)
with c1:
    sel_node = st.selectbox(
        "Filter: Node",
        ["(all)"] + sorted(df["sensor_node_id"].dropna().unique().tolist())
        if not df.empty
        else ["(all)"],
    )
with c2:
    sel_status = st.selectbox(
        "Filter: Latency status", ["(all)", "normal", "delayed", "obsolete"]
    )
with c3:
    sel_type = st.selectbox(
        "Filter: Sensor type", ["(all)", "acoustic", "vision", "hybrid"]
    )

df_f = df.copy()
if not df_f.empty:
    if sel_node != "(all)":
        df_f = df_f[df_f["sensor_node_id"] == sel_node]
    if sel_status != "(all)":
        df_f = df_f[df_f["latency_status"] == sel_status]
    if sel_type != "(all)":
        df_f = df_f[df_f["sensor_type"] == sel_type]

# --- Phase F Patch --- split detail views into Events vs. Tracks tabs
events_tab, tracks_tab = st.tabs(["Events", "Tracks"])

with events_tab:
    st.subheader("Map")
    colA, colB = st.columns(2)
    with colA:
        corridor_width = st.slider("Bearing corridor ±(deg)", min_value=5, max_value=60, value=20, step=1)
    with colB:
        corridor_len_m = st.slider("Corridor length (m)", min_value=100, max_value=1500, value=500, step=50)

    render_map(
        df_f,
        draw_corridor=True,
        half_width_deg=float(corridor_width),
        radius_m=float(corridor_len_m),
    )

    st.subheader("Latency distribution & Node p95")
    lat_ms = latency_ms_series(df_f)
    col1, col2 = st.columns([2, 1])
    with col1:
        if len(lat_ms) > 0:
            bins = st.slider("Latency bins (ms)", 10, 200, 50, step=10)
            hist = pd.cut(lat_ms, bins=bins).value_counts().sort_index()
            # Convert IntervalIndex -> string labels so Streamlit/Altair can render
            hist_df = hist.rename_axis("bin").reset_index(name="count")
            hist_df["bin"] = hist_df["bin"].astype(str)
            st.bar_chart(hist_df.set_index("bin"), use_container_width=True)
        else:
            st.info("No latency data in view.")
    with col2:
        p95_tbl = p95_by_node(df_f)
        st.dataframe(p95_tbl, use_container_width=True, height=280)

    st.subheader("Node health (last seen)")
    health = node_health(df_f, lookback_min=5)
    st.dataframe(health, use_container_width=True)
    warn = health[health["status"] == "WARN"]["sensor_node_id"].tolist() if not health.empty else []
    if len(warn):
        st.warning(f"Nodes missing activity ≥5min: {', '.join(warn)}")
    else:
        st.success("All nodes active within 5 minutes.")

    st.subheader("Recent events")
    if not df_f.empty:
        df_f = df_f.assign(
            latency_ms=df_f["latency_ns"].apply(ns_to_ms),
            created_at=df_f["created_at"].astype(str),
        )
        styled_events = style_events_table(df_f)
        st.dataframe(styled_events, use_container_width=True)
    else:
        st.dataframe(df_f, use_container_width=True)

with tracks_tab:
    # --- Phase F Patch --- display active track rollup alongside contributors
    tracks_df = load_tracks(db_path)
    st.subheader("Active tracks")
    st.dataframe(tracks_df, use_container_width=True)

    contributors_df = load_track_contributors(db_path)
    st.subheader("Track contributors")
    st.dataframe(contributors_df, use_container_width=True)

    st.subheader("Track map")
    if not tracks_df.empty:
        track_map_df = tracks_df.rename(
            columns={
                "aggregated_lat": "lat",
                "aggregated_lon": "lon",
                "aggregated_bearing_deg": "bearing_deg",
            }
        ).copy()
        track_map_df["event_id"] = track_map_df["track_id"]
        track_map_df["sensor_node_id"] = track_map_df["method"]
        track_map_df["latency_ns"] = (
            track_map_df["last_ts_ns"] - track_map_df["first_ts_ns"]
        )
        track_map_df["latency_status"] = track_map_df["status"]
        render_map(track_map_df)
    else:
        st.info("No active tracks available for mapping yet.")
