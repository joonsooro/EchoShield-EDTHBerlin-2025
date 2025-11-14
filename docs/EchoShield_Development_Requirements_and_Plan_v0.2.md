# EchoShield – Development Requirements & Plan

> Version: **v0.2** – Internal Product Engineering Use
> Target: MVP Phase for Edge Acoustic Detection with readiness for Vision/Fusion and Multi-Edge Aggregation

## 1. Background & Scope

**Background:**
EchoShield aims to provide distributed edge-sensor (acoustic/vision) detection of drone threats and rapid transmission of alerts into a C2 chain under low-bandwidth/low-power conditions. From its inception, the system must be designed to extend to vision sensors, multi-edge aggregation (infantry nodes, vehicle nodes), and full sensor-fusion.

**Scope for MVP and Near-Term Plan:**

* Implement the **Wire Packet Schema** (edge format) and **Canonical Schema** (C2/ingestion format).
* Build **Latency Diagnostics** capability, **Event Deduplication & Validity** logic, and a baseline **KPI Metrics** module.
* Architect the system for **Fusion Layer** ready state, and enable **Multi-Edge Aggregation** logic for multiple edge nodes.
* Ensure architecture supports mass deployment (infantry + vehicles) and aggregated detection logic across nodes.

## 2. Development Requirements

### 2.1 Wire Packet Schema (Edge)

**Requirement:** Define and implement a compact edge-message schema (.proto) sized around ~40 bytes for baseband transmission.
**Schema snippet:**

```proto
syntax = "proto3";
package EchoShield;
enum SensorType {
  SENSOR_UNKNOWN = 0;
  SENSOR_ACOUSTIC = 1;
  SENSOR_VISION = 2;
  SENSOR_HYBRID = 3;
}
enum LocationMethod {
  LOC_UNKNOWN = 0;
  LOC_BEARING_ONLY = 1;
  LOC_ACOUSTIC_TRIANGULATION = 2;
  LOC_SENSOR_FUSION = 3;
}
message LocationInt {
  int32 lat_int = 1;       // e.g., degrees * 1e5
  int32 lon_int = 2;
  uint16 error_radius_m = 3;
}
message ContributorInfo {
  string sensor_node_id = 1;
  int64 ts_ns = 2;
  uint16 bearing_deg = 3;
  uint8 bearing_confidence = 4;
}
message WirePacket {
  string event_id = 1;
  SensorType sensor_type = 2;
  uint64 ts_ns = 3;                 // detection timestamp
  uint64 rx_ns = 4;                 // reception timestamp
  LocationInt location = 5;
  uint16 bearing_deg = 6;           // 0-360°, scaled
  uint8 bearing_confidence = 7;     // 0-255 scale
  uint8 n_objects_detected = 8;
  uint8 event_code = 9;
  string sensor_node_id = 10;
  uint8 equipment_type_status = 11;   // 0=unknown,1=assumed,2=classified
  uint8 unit_id_status = 12;          // similar status code
  LocationMethod location_method = 13;  // NEW: method used by the node/packet
  repeated ContributorInfo contributing_edges = 14;  // NEW: optional list of nodes contributing
}
```

**Implementation notes:**

* If the device cannot compute position and only provides direction, set `location_method = LOC_BEARING_ONLY`; populate `bearing_deg` + `bearing_confidence`; keep `location.error_radius_m` coarse (e.g., ≥50–100 m) or leave lat/lon as sensor deployment coords with large error radius.
* If the device has a dual-mic (or array) and can estimate intra-node TDOA to triangulate locally, set `location_method = LOC_ACOUSTIC_TRIANGULATION`. Otherwise rely on multi-edge aggregation upstream.
* Encoding/decoding must be consistent across all edge nodes.
* Versioning mechanism for backward compatibility.
* Edge firmware must capture `ts_ns`; gateway may assign `rx_ns` if needed.

### 2.2 Canonical Schema (C2 Ingestion & Data Store)

**Requirement:** Implement the canonical data model that ingests WirePackets, supports latency/status fields, multi-edge aggregation fields, and future sensor fusion fields.
**JSON snippet:**

```json
{
  "event_id": "uuid-string",
  "sensor_type": "acoustic" | "vision" | "hybrid",
  "ts_ns": 1234567890123456,
  "rx_ns": 1234567891123456,
  "latency_ns": 1000,
  "latency_status": "normal" | "delayed" | "obsolete",
  "location": { "lat": 52.520000, "lon": 13.405000, "error_radius_m": 30.0 },
  "bearing_deg": 235.0,
  "bearing_confidence": 0.78,
  "n_objects_detected": 1,
  "event_code": "DRONE_DET",
  "sensor_node_id": "NODE_A04",
  "unit_id": null,
  "unit_id_status": "unknown",
  "equipment_type": null,
  "equipment_type_status": "unknown",
  "equipment_type_confidence": 0.0,
  "equipment_candidate_list": [],
  "optional_activity_code": null,
  "sensor_metadata": {
    "sensor_deployment_lat": 52.5205,
    "sensor_deployment_lon": 13.4055,
    "sensor_orientation_azimuth_deg": 120.0,
    "sensor_orientation_error_deg": 5.0,
    "sensor_health_status": "nominal",
    "microphone_array_config": null,
    "node_geometry_baseline_m": null
  },
  "classification_history": [],
  "contributing_edges": [],
  "aggregated_location_lat": null,
  "aggregated_location_lon": null,
  "aggregation_confidence": null,
  "object_track_id": null,
  "location_method": "bearing_average" | "acoustic_triangulation" | "sensor_fusion",
  "remarks": "Initial acoustic detection, vision follow-up pending"
}
```

**Implementation notes:**

* If `location_method = bearing_average` or `LOC_BEARING_ONLY` at ingest, do **not** promote location to “trusted” until:

  * multi-edge aggregation provides an `aggregated_location_*`, **or**
  * a vision confirmation arrives.
* For bearing-only events, set/retain a **larger** `location.error_radius_m` and compute a **bearing corridor** for UI (e.g., ±T_bearing degrees).

### 2.3 Latency Diagnostics Capability

* Compute `latency_ns = rx_ns − ts_ns`.
* Derive `latency_status`: e.g., p95 ≤ 500 ms → “normal”; 500 ms < latency ≤ 2 s → “delayed”; latency > 2 s → “obsolete”.
* Dashboard shows p50/p95/p99 per node and overall.
* Logic to lower weight of detections from nodes persistently exceeding latency thresholds.

### 2.4 Event Deduplication & Validity Policy

* In bearing-only mode, increase reliance on (Δt, node proximity, bearing similarity) and **defer hard spatial checks** (Δr) until triangulated or vision-confirmed location exists.
* Same-object criteria: Δt ≤ 5 s, Δbearing ≤ 20°, confidence ≥ 0.6 (use Δr once location exists).
* Validity: `bearing_confidence` ≥ 0.5, `latency_status` ≠ “obsolete”, node health nominal.
* Lifecycle: new → update → expired after 120 s.
* Metrics: duplicate_rate ≤ 5%, validity pass_rate ≥ 85%.

### 2.5 KPI Metrics Definition

* “Bearing-Only Track Promotion Time” — avg. time from first bearing-only event to triangulated/vision-confirmed location (target: ≤ 10 s).
* “Bearing Corridor Hit-Rate” — % of ground-truth positions within `bearing_deg ± T_bearing` corridor (baseline set in pilot).
* Detection Rate ≥ 90%, False Alarm Rate ≤ 5%.
* Latency p95 ≤ 500 ms; Latency Status “Normal” for ≥ 80% of events.
* Bandwidth per event ≤ 80 bytes.
* Node health nominal rate ≥ 90%.
* Operator alert load ≤ 10 per 100 events.

### 2.6 Fusion Layer Architecture & Multi-Edge Aggregation Logic

* Implement **Bearing-Only Association**: cluster bearing-only detections by time window and angular proximity; hold candidates in a short sliding window to await a second bearing or vision confirmation before alert promotion.
* Sensor Adapter Module, Fusion Engine, Multi-Edge Aggregation remain as defined.

## 3. Development Plan & Timeline

| Phase                                           | Duration | Deliverables                                                                 |
| ----------------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| Phase 0 – Requirements & Architecture           | 2 weeks  | Final schema definitions, architecture document v0.1 → v0.2, dev backlog     |
| Phase 1 – Edge Wire Packet Implementation       | 3 weeks  | Edge firmware update, schema encoding/decoding test                          |
| Phase 2 – C2 Ingestion & Canonical Data Store   | 3 weeks  | Ingest API, DB schema, latency diagnostics instrumentation                   |
| Phase 3 – Event Deduplication & Validity Module | 2 weeks  | Dedup logic, validity states, metrics logging                                |
| Phase 4 – KPI Dashboard & Baseline Trial        | 2 weeks  | Dashboard build, baseline trials with nodes, metric gathering                |
| Phase 5 – Multi-Edge Aggregation Preparation    | 3 weeks  | Candidate association logic prototype, schema updates for aggregation        |
| Phase 6 – Fusion Layer Readiness (Vision-Ready) | 4 weeks  | Fusion Engine stub, sensor adapter readiness, vision module integration plan |
| Phase 7 – Field Pilot & Metrics Review          | 2 weeks  | Deploy nodes (infantry + vehicle), gather data, review KPI, roadmap update   |

**NEW Tests:**

* (Phase 5) Bearing-only clustering test with ≥2 nodes; verify candidate promotion when second bearing arrives.
* (Phase 7) Measure “Bearing-Only Track Promotion Time” and “Bearing Corridor Hit-Rate”.

## 4. Resources & Responsibilities

* **Algorithm/Fusion Team:** **NEW:** implement bearing-only association logic (time-window + angular clustering), set default corridor width (e.g., ±20°) adjustable per theatre.

## 5. Risks & Mitigation

* **NEW Risk:** Prolonged bearing-only state due to sparse node coverage → delayed promotion.

  * **Mitigation:** Tune window size, allow corridor-based “advisory alerts,” prioritize adding a second node/vision confirmation in coverage planning.

## 6. Success Criteria for MVP

* Bearing-only streams successfully cluster and promote to triangulated or vision-confirmed tracks in ≤ 10 s (pilot target).
* Bearing Corridor Hit-Rate baseline established and tracked on KPI dashboard.
* End-to-end chain: edge wire packet generation → transport → ingestion → canonical schema ingestion working.
* Latency measurement working: `latency_ns`, `latency_status` fields correctly generated, dashboard visible.
* Deduplication & validity logic operating, with duplicate rate ≤ 10% and validity pass rate > 70% in trial.
* Multi-edge aggregation logic prototype implemented: `contributing_edges` list, `object_track_id` assignment, aggregated_location lat/lon present.
* KPI dashboard built and baseline data collected.

## 7. Next Steps

* Define default **bearing corridor width** and **time-window** for clustering (e.g., ±20°; 5–10 s), document as tunables in config.
* Circulate this document to development lead, firmware team lead, data/analytics lead and fusion algorithm lead for review by **[Date: YYYY-MM-DD]**.
* Set up project board (Jira/Trello) with epics for each phase, assign owners and target dates.
* Begin Phase 0 kick-off meeting to confirm schemas, architecture, meta dependencies (time sync, node metadata).
* Prepare wire packet schema and canonical schema documents as deliverables for Phase 1.
* Plan field trial logistics in parallel: identify node types (infantry, vehicle), comms environment, bandwidth simulation.

---

*Document version: v0.2a – For internal product engineering use only.*
