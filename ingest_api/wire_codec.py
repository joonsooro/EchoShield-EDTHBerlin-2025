
# ingest_api/wire_codec.py
from __future__ import annotations

from typing import Dict, Tuple

from .schemas import CanonicalEvent, WirePacketIn


def _lat_lon_from_ints(location: Dict[str, int]) -> Tuple[float, float, float]:
    lat = location.get("lat_int", 0) / 1e5
    lon = location.get("lon_int", 0) / 1e5
    err = float(location.get("error_radius_m", 5000))
    return lat, lon, err

def _latency_status(latency_ns: int) -> str:
    # thresholds: p95 <= 500ms normal; <=2s delayed; else obsolete
    ms = latency_ns / 1_000_000
    if ms <= 500:
        return "normal"
    if ms <= 2000:
        return "delayed"
    return "obsolete"

def to_canonical(wire: WirePacketIn, rx_ns: int) -> CanonicalEvent:
    lat, lon, err = _lat_lon_from_ints(wire.location.dict())
    bearing_deg = None
    if wire.bearing_deg is not None:
        bearing_deg = float(wire.bearing_deg) / 100.0

    bearing_conf = float(wire.bearing_confidence) / 100.0

    latency_ns = max(0, rx_ns - int(wire.ts_ns))

    # Pass through GCC-PHAT metadata if available
    gcc_phat_metadata = None
    if wire.gcc_phat_metadata is not None:
        gcc_phat_metadata = wire.gcc_phat_metadata.dict()

    return CanonicalEvent(
        event_id=wire.event_id,
        sensor_type=wire.sensor_type,  # normalized in validator
        ts_ns=int(wire.ts_ns),
        rx_ns=rx_ns,
        latency_ns=latency_ns,
        latency_status=_latency_status(latency_ns),
        location={
            "lat": lat,
            "lon": lon,
            "error_radius_m": err
        },
        bearing_deg=bearing_deg,
        bearing_confidence=bearing_conf,
        n_objects_detected=int(wire.n_objects_detected),
        event_code=str(wire.event_code),
        sensor_node_id=wire.sensor_node_id,
        location_method=wire.location_method or "LOC_BEARING_ONLY",
        packet_version=wire.packet_version or 1,
        gcc_phat_metadata=gcc_phat_metadata,
    )
