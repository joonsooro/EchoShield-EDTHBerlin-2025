# Project Structure — EchoShield (v0.1)

This document describes the directory layout, purpose of each folder, and conventions for the `echoshield` repository.
It is intended to support scalability, modularity and clean separation between Edge, Adapter, Core and UI layers.

```
echoshield/
  edge_webapp_adapter/
    index.html
    edge_artifacts/
      feature_norm_and_config.json
      drone_33d_mlp.onnx
    app.v4.js
    app.legacy.js
    config.yaml
    main.py
    mappers.py
    tests/
  ingest-api/
    app.py
    schemas.py
    wire_codec.py
    dedup_validity.py
    kpi.py
    bus.py
    settings.py
    tests/
  c2-core/
    aggregator.py
    fusion_stub.py
    tracks.py
    tests/
  store/
    models.sql
    dao.py
    migrations/
  ui/
    app.py
    components/
  docs/
    API_contract.md
    PROJECT_STRUCTURE.md
    OPERATIONS.md
  Makefile
  .env.example
  README.md
```

## Folder descriptions

- **edge_webapp_adapter/**
  Houses the full “edge‑deployed web app” code (front‑end) plus the adapter logic that transforms client alerts into the canonical *WirePacket* format and forwards it to the ingest API.
  - `index.html`, `app.v4.js`, `app.legacy.js` — Browser UI and inference logic.
  - `edge_artifacts/` — Model and configuration files used by the web app.
  - `config.yaml` — Runtime configuration (e.g., `INGEST_URL`, `NODE_ID`, `ALERT_THRESHOLD`).
  - `main.py`, `mappers.py` — Adapter service logic (FastAPI) to receive web alerts and forward them correctly.
  - `tests/` — Unit/Integration tests for adapter functionality.

- **ingest-api/**
  Core ingestion API module. Receives *WirePacket* via `/api/v0/ingest/wire`, converts into canonical event, applies latency/dedup/validity logic, publishes into internal bus.
  - `app.py` — FastAPI endpoints.
  - `schemas.py`,`wire_codec.py` — Data models, serialization logic.
  - `dedup_validity.py` — Deduplication & Validity module.
  - `kpi.py` — Metric collection logic.
  - `bus.py` — In‑process pub/sub abstraction (placeholder for future Kafka/NATS).
  - `settings.py` — Configuration and environment management.
  - `tests/` — Unit tests covering ingest logic.

- **c2-core/**
  Sensor‑fusion and multi‑edge aggregation logic module. Current MVP supports bearing‑only clustering; future extension for vision/hybrid sensors.
  - `aggregator.py` — Candidate clustering and track promotion logic.
  - `fusion_stub.py` — Stub for future fusion engine.
  - `tracks.py` — Domain logic for tracks management.
  - `tests/` — Unit tests for aggregator and track management.

- **store/**
  Persistence layer (SQLite for hackathon, planner for Postgres).
  - `models.sql` — DDL definitions for `events`, `tracks`, `kpi_rollup`, `nodes`.
  - `dao.py` — Data access object layer.
  - `migrations/` — Placeholder for Alembic or other migration tool as system evolves.

- **ui/**
  Visualization layer built with Streamlit. Shows recent events, KPI metrics, map visualizations of bearings/corridors, tracks summary.
  - `app.py` — Entry point.
  - `components/` — Reusable UI elements (KPI cards, map widgets, tables).

- **docs/**
  Project documentation.
  - `API_contract.md` — Protocol definition (WirePacket, canonical schema, API endpoints).
  - `PROJECT_STRUCTURE.md` — This file.
  - `OPERATIONS.md` — Runbooks, deployment instructions, local dev setup.

- **Makefile**
  Defines common tasks, e.g., `make run-api`, `make run-ui`, `make run-adapter`, `make test-all`.

- **.env.example**
  Example environment variables for local development (database URL, ingest URL, ngrok token etc.).

- **README.md**
  High‐level project overview, quick start instructions, hackathon goals, branch strategy.

## Conventions & Guidelines

- Use **`feature/` or `bugfix/`** prefixes for Git branches.
- All tests should be placed under the module’s `tests/` folder and must pass before merging into `main`.
- Use `black` + `flake8` for Python code formatting/quality. VS Code settings should enforce this.
- Do not mix adapter logic inside `ingest-api` or `c2-core`. Adapter must remain isolated for modularity.
- Keep the WirePacket → Canonical contract stable. Only additive changes allowed unless explicitly versioned.
- Configuration (URLs, thresholds) should live in `config.yaml` or environment variables—not hardcoded in logic.
- Logging in adapter/ingest-api must record at least: timestamp, node_id, event_id, status (‘accepted’, ‘duplicate’, ‘forwarded’), latency in ms.

## Next Steps after Hackathon MVP

- Replace SQLite with Postgres and update `dao.py` + migrations.
- Replace in‐process bus with Kafka or NATS in `bus.py` abstraction.
- Extend `c2-core/fusion_stub.py` into full sensor‑fusion engine (vision + acoustic + hybrid).
- Introduce **time_sync** module to support multi‑node TDOA and augment `WirePacket` with `time_sync_method` + `clock_drift_ns_per_sec`.
- Add authentication/authorization (API Key, OAuth, mTLS) between edge nodes & ingest API.
- Document release notes and migration handbook inside `docs/`.

_End of Project Structure Document_
