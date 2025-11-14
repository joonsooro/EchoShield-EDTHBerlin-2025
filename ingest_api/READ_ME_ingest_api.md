
# Phase D — Ingest API & Event Storage

## Run
```bash
# from repo root
python -m venv .venv
source .venv/bin/activate

make install-api
make run-api-dev
# -> http://localhost:8080/health
```

## Test insert (curl)
```bash
curl -s -X POST http://localhost:8080/api/v0/ingest/wire   -H 'Content-Type: application/json'   -d '{
    "event_id":"t-123",
    "sensor_type":"acoustic",
    "ts_ns": 1730640000000000000,
    "sensor_node_id":"NODE_TEST",
    "location":{"lat_int":0,"lon_int":0,"error_radius_m":5000},
    "bearing_deg": null,
    "bearing_confidence": 80,
    "n_objects_detected": 1,
    "event_code": 10,
    "location_method": "LOC_BEARING_ONLY",
    "packet_version": 1
  }'
```

## Inspect DB (sqlite3)
```bash
sqlite3 ingest_api/store/events.db 'select id,event_id,latency_status,lat,lon from events order by id desc limit 5;'
```
