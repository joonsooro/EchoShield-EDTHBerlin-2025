# EchoShield Fusion Layer Architecture Design
Version: v0.1 – Internal Product-Engineering Use

## 1. Design Objective
Design an architecture that supports:
- Low-bandwidth acoustic edge detection (already in place)
- Seamless future integration of vision (or other modality) models
- Sensor fusion that generates actionable alerts via C2 the moment the vision (or other) model is deployed
- Modularity, edge-capability, latency awareness, and C2 integration

## 2. Architectural Layers
### 2.1 Edge Sensing Layer
- Sensors: Acoustic node(s) already deployed; Vision node(s) planned/future
- Local pre-processing: detection, bearing calculation, rough location estimation
- Wire packet generation: compress data for low-bandwidth transmission
- Priority & queue logic: decide what to send when bandwidth/latency constrained
- “Node” refers to a physical edge device. If the device includes a dual-microphone array or dedicated acoustic sensor module, it may compute internal bearing or time-difference of arrival (TDOA). Otherwise localisation will rely on multiple nodes.

### 2.2 Wire Packet Transport Layer
- Protocol: compressed message format (.proto / FlatBuffer) defined from the start
- Metadata header: `sensor_node_id`, `packet_version`, timestamp(s), security/authentication fields
- Transport logic: handle packet queuing, retransmission, delay measurement
- Latency instrumentation: include `ts_ns`, `rx_ns` fields to measure delays

### 2.3 Fusion Ingestion & Processing Layer
- Reception: wire packets decoded into canonical schema
- Storage: insert into DB or data store, persistent records
- Fusion Engine:
  - Initially: acoustic-only data flows/Acoustic-Only Triangulation Mode (supports multi-node acoustic localization)
  - Later: vision (or other sensor) data arrives → engine correlates events, fuses modalities
  - Outputs: refined object detection, classification, track updates, alert generation
- Deduplication & Validity module: ensure no duplicate event_id creation and filter low-quality events
- Update logic: update event records when vision input arrives or fusion triggers

### 2.4 C2 Command & UI Layer
- Real-time alerting: push alerts to operations/user interface the moment fused detection is validated
- Visualization: object tracks, status, node health, latency metrics
- Analytics/historical view: event history, fusion outcomes, node performance, latency distributions
- Feedback loop: operator annotations, model retraining triggers, node maintenance alerts

## 3. Data & Module Flow Scenario
1. Acoustic edge node detects a drone → Acoustic-Only Triangulation Mode (local bearing + rough location computed) → wire packet sent
2. Transport layer receives packet → `rx_ns` set, canonical record created
3. Vision node (when deployed) detects same object → wire packet sent → fusion engine identifies same event_id → updates record → alert issued
4. C2 UI receives the alert, operator views track, issues directive
5. Analytics subsystem logs the entire chain: detection → transmission → fusion → alert → operator action

## 4. Modular / Future-Proof Design Considerations
- Sensor Adapter Module: supports plugging in new sensor types (acoustic, vision, radar, lidar) without redesigning the entire pipeline
- Fusion Engine: built with multi-modality in mind from Day 1 (even if vision not yet deployed)
- Feature Flags / Versioning: pipeline supports turning vision fusion “on” when available, without architecture change
- Latency Monitoring Module: captures `latency_ns`, computes p95/p99, logs `latency_status` for each event
- Scalable Edge Processing: design supports moving more processing to edge to reduce data transfer and latency
- Node metadata must capture microphone array geometry or baseline if acoustic triangulation within node is supported; multi-node geometry must be considered when performing acoustic-only triangulation across nodes.

## 5. Design Validation & Test Plan
- Edge test: acoustic node → simulate detection → measure end-to-end latency to canonical storage
- Fusion test: when vision node integrated → simulate simultaneous acoustic + vision detections → validate correct matching/event_id reuse & alert generation
- Latency scenarios: emulate high band-limit/comm degradation → verify pipeline still functions, alerts or reference mode behave correctly
- Metrics: track latency p95/p99, event validity rate, missed-fusion rate, duplicate event rate
- Review cycle: after each development milestone, review architecture for integration readiness (vision module, new sensor types)

---

*End of document*
