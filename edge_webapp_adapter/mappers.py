import uuid
from typing import Any, Dict, Optional

from gcc_phat_bearing import estimate_bearing_multi_node
from node_registry import get_registry

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

    # Get node registry
    registry = get_registry()

    # Register/update this node's location if GPS available
    if lat_f is not None and lon_f is not None:
        registry.register_node(node, lat_f, lon_f, acc_f)
        registry.add_detection(node, event_id, ts_ns, conf, lat_f, lon_f)

    # Try to estimate bearing using GCC-PHAT with nearby nodes
    gcc_phat_result = None
    location_method = "LOC_BEARING_ONLY"

    if lat_f is not None and lon_f is not None:
        # Find nearby nodes within 100m radius
        nearby_nodes = registry.get_nearby_nodes(node, max_radius_m=100.0)

        if nearby_nodes:
            # Get concurrent detections from nearby nodes
            concurrent_detections = registry.find_concurrent_detections(
                ts_ns,
                time_window_ns=5_000_000_000,  # 5 seconds
                min_confidence=0.5
            )

            # Add timestamp info to nearby nodes for TDOA calculation
            for nearby in nearby_nodes:
                nearby_id = nearby['node_id']
                recent = registry.get_recent_detections(nearby_id, time_window_sec=5.0)
                if recent:
                    # Use most recent detection timestamp
                    nearby['ts_ns'] = recent[-1]['ts_ns']

            # Estimate bearing using GCC-PHAT
            current_node_data = {
                'node_id': node,
                'lat': lat_f,
                'lon': lon_f,
                'ts_ns': ts_ns
            }

            gcc_phat_result = estimate_bearing_multi_node(
                current_node_data,
                nearby_nodes,
                ts_ns
            )

            if gcc_phat_result:
                # Use GCC-PHAT bearing instead of single-node bearing
                bearing_deg = gcc_phat_result['bearing_deg']
                bearing_conf_pct = int(round(gcc_phat_result['bearing_confidence'] * 100))
                location_method = "LOC_ACOUSTIC_TRIANGULATION"

    # Build WirePacket
    wire_packet = {
        "event_id": event_id,
        "sensor_type": SENSOR_TYPE,
        "ts_ns": ts_ns,
        "sensor_node_id": node,
        "location": loc,
        "bearing_deg": None if bearing_deg is None else int(round(float(bearing_deg) * 100)),
        "bearing_confidence": bearing_conf_pct,
        "n_objects_detected": 1,
        "event_code": evt_code,
        "location_method": location_method,
        "packet_version": 1,
    }

    # Add GCC-PHAT metadata if available
    if gcc_phat_result:
        wire_packet["gcc_phat_metadata"] = {
            "method": gcc_phat_result['method'],
            "paired_node_id": gcc_phat_result['paired_node_id'],
            "baseline_distance_m": round(gcc_phat_result['baseline_distance_m'], 2),
            "tdoa_sec": round(gcc_phat_result['tdoa_sec'], 6),
            "baseline_bearing_deg": round(gcc_phat_result['baseline_bearing_deg'], 2)
        }

    return wire_packet
