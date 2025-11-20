MAIN BRANCH NOT UP TO DATE. "berlin" BRANCH IS CURRENTLY THE MAIN BRANCH FOR TECHNICAL UPDATES

# EchoShield — Passive Acoustic ISR Node (EDTH Berlin 2025 MVP)

**Document Type:** Tactical Product & System Overview  
**Audience:** Helsing PM / Perception / Platform teams  
**Status:** MVP Prototype (EDTH Berlin Finalist)  
**Owner:** Rudin Ro  
**Version:** v1.0  

---

## 1. Purpose

EchoShield is a passive, edge-deployed acoustic ISR node designed for early bearing-based drone awareness in environments where EM emissions must remain silent and RF sensors may be limited or degraded.

This document describes the actual implemented behavior of the EDTH Berlin MVP:
- What exists and runs today  
- What constraints shaped the design  
- What decisions were made and why  
- What is explicitly out of scope  
- How the MVP demonstrates tactical viability  

Only implemented functionality is included.

---

## 2. System Overview

### On-Device Acoustic Node (iPhone Safari)
- Real-time audio capture  
- Lightweight drone classifier (ONNX MLP, 33-D handcrafted features)  
- Bearing estimation using GCC-PHAT  
- Local heading correction  
- Event microping POST to edge adapter (~50–80 bytes)

### Edge WebApp Adapter (FastAPI)
- Receives detection events  
- Normalizes payload into internal WirePacket schema  
- Stores events to SQLite  
- Optional upstream forwarding

### Mock C2 Fusion Layer
- Multi-node time-window alignment  
- Bearing aggregation  
- Triangulation via 2D least-squares  
- Basic smoothing/tracking placeholder (non-functional)

This delivers a complete detection → event → C2 alert → triangulation loop.

---

## 3. Architecture

[On-Device Node]
• Audio capture
• Feature extraction (33D)
• ONNX inference
• GCC-PHAT DOA (azimuth)
• Heading correction
→ POST microping

      ▼

[Edge WebApp Adapter]
• FastAPI /webhook/edge
• WirePacket normalization
• SQLite persistence

      ▼

[C2 Fusion Layer]
• NodeDetection fetch
• Time-window grouping
• Triangulation (Least Squares)
• Stub smoothing

      ▼

[Operator UI]
• Node layout
• Bearings
• Fused point
• KPIs (dup %, contributors, latency)

---

## 4. Validated Capabilities (Implemented)

- On-device ONNX MLP drone classifier  
- GCC-PHAT bearing estimation  
- Local → global azimuth conversion  
- Microping-based data transmission (<80 bytes)  
- Multi-node alignment using rolling time window  
- Least-squares triangulation  
- Functional end-to-end detection → edge → C2 pipeline  
- End-to-end latency ~500–650 ms depending on device  

All were validated during EDTH Berlin test runs.

---

## 5. System Constraints (Observed Only)

### Device / Environment
- iPhone mic noise suppression affects detection range  
- Safari throttles JS when backgrounded  
- Device mic geometry varies (heading parameter required)

### Architectural
- No cross-node clock sync: fusion relies on window-based alignment  
- Single-target assumption  
- No worker-thread offload for DOA (Safari restriction)

### ML / DSP
- No adaptive filtering  
- No SNR-aware logic  
- No orientation-agnostic DOA

---

## 6. Known Limitations

| Area                      | Impact | Notes |
|--------------------------|--------|-------|
| No clock sync            | High   | Limits long-range triangulation accuracy |
| Safari throttling        | Medium | Affects inference stability |
| Mic geometry variance    | Medium | Requires heading input |
| Single-target assumption | Medium | Multi-target unsupported |
| No DOA smoothing         | Low    | Basic LS triangulation only |

---

## 7. Performance (Measured)

**Node-local latency:**  
~50 ms (audio callback → inference → DOA → POST)

**End-to-end latency:**  
~500–650 ms (Node → Edge → DB → Fusion → UI)

Values are based on measured runs during EDTH Berlin.

---

## 8. MVP Scope

### Included
- ONNX MLP classifier  
- GCC-PHAT DOA  
- Multi-node LS triangulation  
- SQLite event storage  
- Time-window fusion  
- Streamlit UI
- Model Quantization (TinyML)

### Out of Scope
- Multi-target tracking  
- RF/vision fusion  
- Orientation-agnostic DOA  
- WebGPU acceleration  
- Range estimation  
- Smoothing filters (non-functional stub only)

---

## 9. Next Steps (Roadmap, Not Implemented)

- Add DOA smoothing for bearing stability  
- Introduce worker-thread DSP for Safari-safe background behavior  
- Add basic multi-target logic  
- Improve heading reliability (IMU or compass input)  
- Harden event schema for C2 interoperability  

All items are future pathway suggestions, not current functionality.

---

## 10. Summary

EchoShield (EDTH Berlin MVP) demonstrates a functioning passive acoustic ISR pipeline:

- On-device detection  
- Bearing estimation  
- Microping transport  
- Multi-node alignment  
- Real triangulation  
- Tactical-range latency performance  

Not production-ready, but provides a clear, credible path toward a distributed passive ISR system suitable for further development.

---
