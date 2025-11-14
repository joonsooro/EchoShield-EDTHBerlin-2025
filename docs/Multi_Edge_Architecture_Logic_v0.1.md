# Multi-Edge Aggregation Logic – EchoShield
> Version: v0.1 – Internal Product-Engineering Use

## 1. Purpose
- Aggregate detections from multiple edge devices (infantry-field, vehicle-mounted, static nodes) to infer a more accurate location and classification of a common object (e.g., drone).
- Improve accuracy, reduce false alerts, and provide consistent object tracking across heterogeneous sensors and nodes deployed broadly.

## 2. Logic Overview
### 2.1 Detection Receipt & Candidate Initialization
- Each edge node sends a Wire Packet upon detection → converted to Canonical Schema record.
- A detection candidate object is created containing: `event_id`, `sensor_node_id`, `location`, `bearing`, `confidence`, `timestamp`.

### 2.2 Candidate Association (Same-Object Determination)
When a new detection arrives, assess whether it belongs to an existing candidate object using:
- Time delta Δt ≤ T_time_assoc (e.g., 5 s)
- Spatial delta Δr between locations ≤ T_dist_assoc (e.g., 30–50 m)
- Bearing difference |Δbearing_deg| ≤ T_bearing_assoc (e.g., ±20°)
- Confidence new_detection ≥ T_confidence_assoc (e.g., 0.6)
- Sensor node proximity/affiliation considered
If criteria met, append detection to existing candidate’s `contributing_edges[]`; else new candidate.

### 2.3 Aggregation & State Estimation
- Once a candidate meets either: number of contributing edges ≥ N_min or elapsed time ≥ T_window → trigger aggregation.
- Aggregation algorithm may follow:
  - Weighted average of lat/lon (weights by 1/error_radius_m and/or bearing_confidence)
  - Bayesian or Kalman filter to compute refined object state (location, velocity, heading)
  - Aggregated confidence = f(number_of_contributors, individual confidence values)
  - Assign `object_track_id` and update Canonical Schema with: `aggregated_location_lat`, `aggregated_location_lon`, `aggregation_confidence`, `contributing_edges_count`

### 2.4 Alert Generation & C2 Update
- When `aggregation_confidence` ≥ T_alert_confidence → mark event as actionable and send to C2 UI.
- Update record with `fusion_stage` = “multi-edge”, `duplicate_flag` = false, `object_track_id`.
- If detection was previously separate events, mark them `duplicate_flag` = true and link to same `object_track_id`.

### 2.5 Track Lifecycle & Maintenance
- For each object_track_id, maintain `track_history[]` entries: timestamp, location, confidence, contributing_edges list.
- If no new detection for the object within T_expiry seconds → mark track as expired.
- After expiry, new detections create a new `object_track_id`.
- Maintain analytics: number_of_contributors, mean latency among contributors, error_variance, track_duration.

## 3. Parameter Recommendations
| Parameter               | Suggestion     | Notes                                  |
|-------------------------|----------------|----------------------------------------|
| T_time_assoc            | 5 s            | Based on typical acoustic-to-vision gap |
| T_dist_assoc            | 30 m           | Considering acoustic location error    |
| T_bearing_assoc         | ±20°           | Allows bearing deviation tolerance     |
| T_confidence_assoc      | 0.60           | Minimum confidence to join aggregation |
| N_min                   | 2 devices      | Require at least two contributing nodes|
| T_window                | 10 s           | Maximum window waiting for contributors|
| T_expiry                | 120 s          | Track expiry threshold                 |
| T_alert_confidence      | 0.75           | Confidence threshold to generate alert |

## 4. Technical & Operational Considerations
### 4.1 Time Synchronization
- Edge nodes must maintain reliable timestamp (`ts_ns`). Recommend GPS synchronization or PPS/NTP fallback.
- Include node clock offset metadata and compensate during aggregation.

### 4.2 Node Error & Confidence Modeling
- Each node carries baseline error model (e.g., infantry_deploy_error_m = 50 m) and `bearing_confidence`.
- Aggregation weights individual detections accordingly to improve final estimate.

### 4.3 Latency & Transmission Prioritization
- Include contributors’ `latency_ns` metrics in aggregation. Nodes with high latency may receive lower weight or be excluded.
- Edge node logic should support “summary packet” mode in constrained comms.

### 4.4 Scalability & Volume Control
- Use a sliding window queue for candidate objects and limit contributing edges count to avoid explosion in large-scale deployment.
- Support hierarchical processing: local cluster aggregation → regional fusion hub → C2 global fusion.

### 4.5 Security & Trust
- Packets carry cryptographic signature or node authentication tag.
- Maintain node trust_score: reduce weight for nodes with frequent faults, delays, or errors.

### 4.6 Mobility & Environmental Impact
- Node metadata includes `mobility_status` (static/deployed/moving) and `terrain_category`.
- Adjust error/weight for moving nodes and reflect terrain blockage or NLOS conditions in algorithm.

### 4.7 Node Health & Field Operations
- Maintain `node_health_status_history[]` for each sensor (battery, last_ping, error_rate).
- Dashboard tracks high-delay or high-error nodes for maintenance or redeployment.

## 5. Implementation & Test Plan
1. Develop simulation harness: generate synthetic multi-edge detection data for candidate object → validate association logic.
2. Offline aggregation test: feed multiple detection streams, verify `aggregated_location`, `aggregation_confidence`, `object_track_id` correct assignment.
3. Field pilot: deploy 3-5 edge nodes, generate real detections, verify aggregation with known ground truth.
4. Scale test: emulate 50+ nodes contributing detections, monitor system responsiveness, queue depth, throughput.
5. Parameter tuning: refine thresholds (T_time_assoc, T_dist_assoc, etc) based on pilot results and analytics.

---

*End of document*
