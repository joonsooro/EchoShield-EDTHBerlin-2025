# EchoShield Latency-Agnostic Design Strategy

> **Purpose:** Ensure the system continues to provide meaningful detection and alert capability even in high-latency or variable-latency tactical environments.

## 1. Latency Allowance & Status Definitions
- Define statistical latency thresholds (e.g., p95, p99) for operational modes.
  - Example: p95 ≤ 500 ms → “normal”, 500 ms < latency ≤ 2 s → “delayed”, latency > 2 s → “obsolete”.
- Use the field `latency_ns = rx_ns − ts_ns` to measure end-to-end latency.
- Populate the field `latency_status` (“normal”, “delayed”, “obsolete”) to flag how fresh the information is.

## 2. Freshness / Age of Information (AoI) Concept
- Recognition that the value of detection decays with time.
- Policy example:
  - If `latency_ns` ≤ T1 → mode = **“real-time alert”**
  - If T1 < `latency_ns` ≤ T2 → mode = **“reference/support”**
  - If `latency_ns` > T2 → mode = **“post-event analysis”**
- State transitions must be visible via `latency_status` field and in the operational UI.

## 3. Priority and Transmission Control
- Edge nodes adopt logic for **priority determination** based on `bearing_confidence`, `event_code`, `n_objects_detected`, etc.
- High-priority events are transmitted immediately; low priority may be queued or summarized.
- In constrained or high-latency comms environments, use **summary transmission** or **local pre-processing then uplink**.

## 4. Node-wise Latency Tracking & Metadata Use
- Include `sensor_node_id` and `sensor_metadata` so that latency and performance can be tracked per node.
- If a node consistently exhibits high latency, trigger operator alert or node reposition/update.
- Latency debugging: because each event logs `ts_ns`, `rx_ns`, and `sensor_node_id`, you can identify where delays occur (sensor → transmit → comms → receive).

## 5. State Transitions & Data History Management
- The C2 system schema must include fields like `verification_phase`, `update_count`, `last_updated_ts_ns`.
- Events flagged as “delayed” or “obsolete” should **still be stored**, and the history must be maintained for later analysis.
- The UI/dashboard must support filtering and visuals by `latency_status`, enabling operators to focus on fresh vs stale data.

## 6. Design Verification & Test Plan
- **Latency simulation testing:** Emulate various network/bandwidth/comm-delay scenarios and measure p95/p99 latency.
- **Field trials:** Deploy edge nodes → network → C2 chain, capture and log `ts_ns`, `rx_ns`, and compute `latency_ns`.
- **Operational review cycles:** On a regular schedule, review events by latency status and evaluate their operational value to refine thresholds.
- Key reference: *“Persistent ISR at the Tactical Edge”* and other tactical-edge sensor/AI latency studies.

---

*Document version: v0.1 – for internal product-engineering use. All thresholds (T1, T2) must be customized per operational theater and network environment.*
