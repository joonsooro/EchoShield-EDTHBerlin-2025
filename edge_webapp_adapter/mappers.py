import uuid
from typing import Any, Dict

SENSOR_TYPE = "acoustic"
EVENT_CODE_MAP = {"drone": 10}

def to_wirepacket(payload: Dict[str, Any]) -> Dict[str, Any]:
    event_id = str(uuid.uuid4())
    ts_ns = int(payload.get("ts_ns") or int(float(payload["time_ms"])) * 1_000_000)
    node = str(payload.get("nodeId") or payload.get("node_id") or "NODE_UNKNOWN")
    bearing_deg = payload.get("azimuth_deg", None)
    conf = float(payload.get("confidence", 0.0))
    bearing_conf_pct = max(0, min(100, int(round(conf * 100))))

    # Optional device geolocation passthrough (if provided by webapp)
    # Accept multiple key variants (flat or nested), and tolerant of string numbers.
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _first_float(keys, src):
        for k in keys:
            if isinstance(k, tuple) and len(k) == 2:
                parent, child = k
                v = src.get(parent, {})
                if isinstance(v, dict) and child in v:
                    f = _to_float(v.get(child))
                    if f is not None:
                        return f
            else:
                if k in src:
                    f = _to_float(src.get(k))
                    if f is not None:
                        return f
        return None

    # Candidates for latitude/longitude from various payload shapes:
    lat_f = _first_float(["lat", "latitude", ("location", "lat"), ("location", "latitude")], payload)
    lon_f = _first_float(["lon", "lng", "longitude", ("location", "lon"), ("location", "lng"), ("location", "longitude")], payload)
    # Accuracy candidates (meters)
    acc_f = _first_float(["acc_m", "accuracy", "accuracy_m", ("location", "acc_m"), ("location", "accuracy")], payload)

    # Validate ranges and build fixed-point location
    if lat_f is not None and lon_f is not None and (-90.0 <= lat_f <= 90.0) and (-180.0 <= lon_f <= 180.0):
        lat_int = int(round(lat_f * 1e5))
        lon_int = int(round(lon_f * 1e5))
        if acc_f is not None:
            err_m = int(round(acc_f))
            # clamp to sane bounds
            if err_m < 5:
                err_m = 5
            elif err_m > 5000:
                err_m = 5000
        else:
            err_m = 200
        loc = {"lat_int": lat_int, "lon_int": lon_int, "error_radius_m": err_m}
    else:
        # fallback when device geolocation is unavailable/invalid
        loc = {"lat_int": 0, "lon_int": 0, "error_radius_m": 5000}

    evt_name = str(payload.get("event", "drone")).lower()
    evt_code = EVENT_CODE_MAP.get(evt_name, 10)

    return {
        "event_id": event_id,
        "sensor_type": SENSOR_TYPE,
        "ts_ns": ts_ns,
        "sensor_node_id": node,
        "location": loc,
        "bearing_deg": None if bearing_deg is None else int(round(float(bearing_deg) * 100)),
        "bearing_confidence": bearing_conf_pct,
        "n_objects_detected": 1,
        "event_code": evt_code,
        "location_method": "LOC_BEARING_ONLY",
        "packet_version": 1,
    }
