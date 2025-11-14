# KPI Definition Document – EchoShield v0.1
> **Purpose:** Establish objective, quantifiable criteria to evaluate whether this product is successful—in detection → delivery → trust → action workflow.

## 1. Metric Categories
Metrics are grouped into four categories: Detection Performance, Transmission/Latency Performance, Reliability, and Operational Efficiency.

### 1.1 Detection Performance
| Metric Name | Definition | Target Value | Notes |
|------------|------------|-------------|-------|
| Detection Rate | Proportion of actual threat objects (e.g., drones) detected by the system | ≥ 90% | Baseline will adjust after field trials |
| False Alarm Rate (FAR) | Proportion of non-threat items reported as detection events | ≤ 5% | Reducing operator burden is critical |
| Location Error Mean | Average distance error between the system’s reported location (edge + vision) and actual location | ≤ 15-30 m | Considering acoustic location error range |
| Bearing Confidence Mean | Average of `bearing_confidence` values | ≥ 0.75 | Higher direction confidence improves fusion utility |

### 1.2 Transmission & Latency Performance
| Metric Name | Definition | Target Value | Notes |
|------------|-------------|-------------|-------|
| Latency p95 | 95th percentile value of `latency_ns = rx_ns − ts_ns` | ≤ 500 ms | Goal for low-latency environment |
| Latency Status “Normal” Ratio | Ratio of events whose `latency_status = “normal”` | ≥ 80% | Tactical environment realistic threshold |
| Bandwidth Per Event | Average bytes transmitted per event | ≤ 80 bytes | Design constraint for low-bandwidth edge |

### 1.3 Reliability
| Metric Name | Definition | Target Value | Notes |
|------------|-------------|-------------|--------|
| Event Validity Pass Rate | Ratio of events judged “valid” by the system and forwarded to C2 | ≥ 85% | Ties to deduplication & validity policy |
| Node Health Nominal Rate | Ratio of sensor nodes with `sensor_health_status = “nominal”` | ≥ 90% | Ensures sensor-node reliability |
| Duplicate Event Rate | Proportion of excessive duplicate event_id generation for same object | ≤ 5% | Prevents duplicate clutter |

### 1.4 Operational Efficiency
| Metric Name | Definition | Target Value | Notes |
|------------|-------------|-------------|--------|
| Uplink Data Volume per Hour | Volume of data uplinked (edge → C2) per hour | ≤ (Events × 80 bytes) + 10% | Monitors within low-bandwidth envelope |
| Operator Alert Load per 100 Events | Number of alerts delivered to operator per 100 events | ≤ 10 | Measures operator fatigue level |
| Update Time to Fusion | Average time from detection → vision confirmation/fusion | ≤ 10 s | Aim for fused multi-sensor response |

---

## 2. Implementation & Application Guidelines
1. Measure and report the above metrics every development sprint (e.g., every 2-4 weeks) and per release.
2. Acquire baseline values via experiments/pilot, then adjust target values accordingly.
3. For any metric shortfall, perform **Root-Cause Analysis** (e.g., node latency, comms bottleneck, sensor error) → conduct improvement actions → track next period metrics.
4. Use these metrics to present to investors or defense evaluation agencies: illustrate quantitatively that the product delivers real mission value.
5. Assign **Priority & Ownership** for each metric, and integrate into real-time or periodic dashboard for Dev, Data & Ops teams.

---

*Document version: v0.1 – For internal product engineering use only.*
