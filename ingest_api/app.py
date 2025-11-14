
# ingest_api/app.py
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import WirePacketIn
from .wire_codec import to_canonical
from fastapi import Response

DB_PATH = os.environ.get("INGEST_DB", str(Path(__file__).parent / "store" / "events.db"))
SQL_PATH = Path(__file__).parent / "store" / "models.sql"

app = FastAPI(title="EchoShield Ingest API", version="0.1")

@app.get("/whoami", include_in_schema=False)
def whoami_ingest():
    return {"service": "ingest_api", "port": 8080}

def _init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

@app.on_event("startup")
def on_startup() -> None:
    _init_db()

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "db": DB_PATH}

@app.post("/api/v0/ingest/wire")
async def ingest_wire(req: Request) -> JSONResponse:
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    try:
        wire = WirePacketIn(**payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"validation error: {e}")

    rx_ns = int(time.time_ns())
    canonical = to_canonical(wire, rx_ns)

    # compute skew (signed is fine; if you insist on non-negative, wrap with max(0, ...))
    clock_skew_ns = int(canonical.rx_ns) - int(canonical.ts_ns)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO events (
            event_id, sensor_type, ts_ns, rx_ns, latency_ns, latency_status,
            lat, lon, error_radius_m, bearing_deg, bearing_conf, n_objects,
            event_code, sensor_node_id, location_method, packet_version,
            clock_skew_ns, raw_wire_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                canonical.event_id,
                canonical.sensor_type,
                canonical.ts_ns,
                canonical.rx_ns,
                canonical.latency_ns,
                canonical.latency_status,
                canonical.location.get("lat"),
                canonical.location.get("lon"),
                canonical.location.get("error_radius_m"),
                canonical.bearing_deg,
                canonical.bearing_confidence,
                canonical.n_objects_detected,
                canonical.event_code,
                canonical.sensor_node_id,
                canonical.location_method,
                canonical.packet_version,
                clock_skew_ns,
                json.dumps(payload),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Build response with GCC-PHAT metadata if available
    response_data = {
        "ok": True,
        "event_id": canonical.event_id,
        "location_method": canonical.location_method,
        "bearing_deg": canonical.bearing_deg,
        "bearing_confidence": canonical.bearing_confidence
    }

    if canonical.gcc_phat_metadata:
        response_data["gcc_phat"] = canonical.gcc_phat_metadata

    return JSONResponse(status_code=202, content=response_data)
