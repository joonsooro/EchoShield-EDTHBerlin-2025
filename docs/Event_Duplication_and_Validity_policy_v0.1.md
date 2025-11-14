# Event Deduplication & Validity Policy – EchoShield
> Version: v0.1 – internal product-engineering use

## 1. Purpose
- Prevent creation of duplicate events caused by the same physical object being detected multiple times by one or more sensors.
- Ensure only valid, trustworthy event records propagate into C2 and fusion layers, reducing false alerts and operator fatigue.
- Enable persistent tracking of the same object (object = e.g., drone) via consistent `event_id` usage across sensor feeds.

## 2. Deduplication Policy
### 2.1 Same-Object Criteria
An incoming detection record will be considered the **same object** (and thus reuse the existing `event_id`) if the following conditions are met:
- **Time delta (Δt)** between previous detection timestamp and current detection ≤ T_time seconds.
- **Spatial delta (Δr)** between previous detection’s location and current location ≤ T_dist meters.
- **Sensor node proximity/affiliation**: either same node or a trusted nearby node.
- **Bearing similarity**: absolute difference of `bearing_deg` ≤ T_bearing_deg degrees.
- **Confidence threshold**: the detection’s confidence ≥ T_confidence.

If **N of M** conditions (configurable) are satisfied, the detection is treated as the same object. Otherwise, a **new event_id** is generated.

### 2.2 Event Validity Criteria
Before ingestion into the C2/fusion layer, each event shall be evaluated for validity:
- `bearing_confidence` (and any other confidence metrics) must be ≥ T_min_confidence.
- If `latency_status` = “obsolete”, event is flagged as low-value or archived rather than used for real-time alerting.
- If the reporting sensor’s `sensor_health_status` = “degraded” or “failed”, validity is reduced (e.g., assign “low” validity status).
- If no follow-up verification (e.g., vision confirmation, fusion update) within a configured timeframe T_followup, the event remains in “provisional” status or is dropped based on policy.
- Filter out repeated rapid detections from the same node within ∆t_noise (very short interval) to suppress sensor noise/false positives.

### 2.3 Event ID Life-Cycle Management
- When first detection occurs → assign new `event_id`.
- If subsequent detection matches same-object criteria → reuse that `event_id`, update record.
- If no detection update for that `event_id` for T_expiry seconds → mark event as **expired**, archive or close record.
- If expired object appears again after expiry → assign **new** `event_id`.

### 2.4 System Flow & Automation
- **Edge module**: implements quick deduplication checks (time + bearing + node) and tags new vs update.
- **C2 ingestion**: applies validity rules, flags `validity_status` (“valid”, “provisional”, “invalid”), and logs deduplication via fields like `duplicate_flag`, `dedup_delta_t`, `dedup_delta_r`.
- **Dashboard/Analytics**: monitor metrics such as deduplication rate, validity pass rate, false alert rate.
- **Logging**: record `event_id`, `sensor_node_id`, `timestamp`, `previous_event_id` (if reused), and deduplication reason code.

## 3. Policy Parameter Recommendations (Initial Defaults)
| Parameter | Suggested Value | Notes |
|-----------|-----------------|--------|
| T_time | 5 seconds | Considering acoustic→vision transition in tactical environment |
| T_dist | 30 meters | Acoustic coarse location error tolerance |
| T_bearing_deg | ±20° | Bearing deviation allowance |
| T_confidence | 0.60 | Minimum detection confidence for same-object candidate |
| T_min_confidence | 0.50 | Minimum confidence to treat event as valid |
| ∆t_noise | 1 second | Suppress repeated detections within too short interval |
| T_followup | 30 seconds | Time allowed for follow-up confirmation (vision/fusion) |
| T_expiry | 120 seconds | Time after which an object is considered no longer active |

> **Note:** These values are starting points — system characterization and field trials must validate and tune these thresholds.

## 4. Governance & Audit
- All deduplication and validity rules must be documented and subject to version control.
- Maintain audit logs of event lifecycle transitions (`new` → `update` → `expired`) and reasons.
- Periodic review (e.g., each sprint or release) of false alert incidents, duplicate event rate, and validity failures.
- Stakeholders: Data Team, Dev Team, Ops Team must agree on parameter tuning as part of integrated product review.

---

*End of document*
