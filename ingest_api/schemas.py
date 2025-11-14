
# ingest_api/schemas.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, validator

SensorType = Literal["acoustic", "vision", "hybrid"]

class LocationInt(BaseModel):
    lat_int: int
    lon_int: int
    error_radius_m: int


class GccPhatMetadata(BaseModel):
    """GCC-PHAT bearing estimation metadata."""
    method: str  # "GCC_PHAT_TDOA"
    paired_node_id: str
    baseline_distance_m: float
    tdoa_sec: float  # Time difference of arrival in seconds
    baseline_bearing_deg: float  # Bearing between node pair


class WirePacketIn(BaseModel):
    event_id: str
    sensor_type: str  # keep string for wire, map to enum upstream
    ts_ns: int
    sensor_node_id: str
    location: LocationInt
    bearing_deg: Optional[int] = None   # deg * 100, or None
    bearing_confidence: int = Field(ge=0, le=100)
    n_objects_detected: int = 1
    event_code: int
    location_method: Optional[str] = None
    packet_version: Optional[int] = 1
    gcc_phat_metadata: Optional[GccPhatMetadata] = None  # NEW: GCC-PHAT bearing data

    @validator("sensor_type")
    def validate_sensor_type(cls, v: str) -> str:
        v2 = v.lower()
        if v2 not in {"acoustic", "vision", "hybrid"}:
            # allow wire values but normalize
            if v2.startswith("sensor_"):
                v2 = v2.replace("sensor_", "")
            if v2 not in {"acoustic", "vision", "hybrid"}:
                raise ValueError("invalid sensor_type")
        return v2

class CanonicalEvent(BaseModel):
    event_id: str
    sensor_type: SensorType
    ts_ns: int
    rx_ns: int
    latency_ns: int
    latency_status: Literal["normal", "delayed", "obsolete"]
    location: Dict[str, Optional[float]]  # {lat, lon, error_radius_m}
    bearing_deg: Optional[float] = None   # degrees
    bearing_confidence: float = 0.0       # 0..1
    n_objects_detected: int = 1
    event_code: str
    sensor_node_id: str
    location_method: Optional[str] = None
    packet_version: Optional[int] = 1

    # NEW: Phase F output fields (Result from post-processing of the server)
    validity_status: Optional[Literal["valid", "invalid", "unknown"]] = "unknown"
    duplicate_flag: Optional[int] = 0
    object_track_id: Optional[str] = None

    # GCC-PHAT bearing estimation metadata
    gcc_phat_metadata: Optional[Dict[str, Any]] = None

    # (optional diagnostics)
    clock_skew_ns: Optional[int] = None
    bearing_std_deg: Optional[float] = None
