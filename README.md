EchoShield — Acoustic ISR Node (EDTH Berlin Prototype)

Document Type: Engineering Design Note (EDN)
Audience: Helsing DSP/Perception/Platform teams
Status: MVP Prototype (EDTH Berlin 2025)
Owner: Rudin Ro & Akash Prakasan Manikyam
Version: v0.9

⸻

1. Purpose

This document describes the current engineering state of the EchoShield acoustic ISR prototype, including module boundaries, implemented functionality, technical constraints, and integration points relevant for Helsing review.

The intent is to provide engineering clarity on:
	•	What exists and is running today
	•	What is stubbed or simplified
	•	Where the design intentionally leaves future integration points
	•	How this prototype could be hardened for Helsing-grade deployment

⸻

2. System Overview

EchoShield is a passive, on-device acoustic ISR node that:
	1.	Runs real-time drone detection ML inference on iPhone Safari
	2.	Computes bearing (azimuth) via GCC-PHAT DOA
	3.	Transmits detections to a local Edge WebApp Adapter (FastAPI)
	4.	Stores events in SQLite
	5.	Feeds a Mock C2 module that performs:
	•	Multi-node detection alignment
	•	Triangulation (least squares)
	•	Track estimation (stubbed β-filter)

The system is a functional multi-node passive detection + localization pipeline.

⸻

3. High-Level Architecture (Technical)

[iPhone WebApp]
  • ONNX MLP (33D)
  • FFT + GCC-PHAT DOA
  • Local heading correction
  • Streaming inference loop
  • POST → /webhook/edge

        ▼

[Edge WebApp Adapter (FastAPI)]
  • Map → WirePacket schema
  • Append row → events.db
  • Optional forward → ingest_url

        ▼

[SQLite: events.db]
  • node_id, ts_ns, lat, lon
  • azimuth_deg, confidence

        ▼

[Mock C2 Fusion Layer]
  • c2_db_adapter.py (DB → NodeDetection)
  • bearing_collect.py (multi-node time alignment)
  • triangulation_ls.py (least squares)
  • (stub) kalman_2d

        ▼

[Streamlit UI]
  • Node positions
  • Bearings
  • Triangulated point
  • KPI: events, duplicate %, avg contributors


⸻

4. Module Inventory (REAL vs STUB)

4.1 WebApp (JavaScript)

Component	Status	Notes
ONNX MLP inference	REAL	33-D handcrafted audio features
Audio capture & ring buffers	REAL	Stable on iOS Safari
FFT per frame	REAL	JS FFT (optimized)
GCC-PHAT DOA	REAL	Matches ground truth in synthetic tests
Local → global azimuth	REAL	Uses node heading param
Worker-thread offload	STUB	Disabled due to Safari constraints
Energy gating for frames	REAL (simple)	Median-threshold gating
Feature-level quantization	STUB	Placeholder for future perf


⸻

4.2 Edge Adapter (FastAPI, Python)

Component	Status	Notes
/webhook/edge endpoint	REAL	Receives 1 event per ML alert
Payload mapping (mappers.py)	REAL	Converts to WirePacket
Storage to events.db	REAL	Each event stored persistently
API key forwarding	REAL	Upstream ingest tested
Schema validation	REAL	Ignores malformed fields
Device geolocation passthrough	REAL	Lat/lon saved directly


⸻

4.3 DSP Engine (Python)

Module	Status	Function
node_output.py	REAL	Normalized per-node bearing format
bearing_collect.py	REAL	Time-window collection + grouping
triangulation_ls.py	REAL	2D least squares (residual scoring)
kalman_2d.py	STUB	Placeholder smoothing filter


⸻

4.4 DB Adapter (Python)

c2_db_adapter.py
	•	REAL: Loads events.db → NodeDetection
	•	REAL: Local XY projection (equirectangular)
	•	REAL: Time filtering & confidence filtering
	•	REAL: Returns BearingSet for fusion

⸻

4.5 Mock C2 Visualization
	•	REAL: Streamlit map rendering
	•	REAL: KPI: event count, duplicate %, contributors
	•	STUB: Multi-track management
	•	STUB: Threat-level scoring

⸻

5. System Performance (Measured)

Node-local latency
	•	< 50 ms end-to-end from audio callback → ML inference complete.
(Measured on iPhone 15 Pro, Safari, WebAudio inference loop)

End-to-end C2 latency
	•	< 650 ms
Includes:
	•	Estimated time for C2 (Sitaware) Integration via Project Q (MAX 600ms)
    •	ML alert → POST → FastAPI
	•	DB append
	•	C2 refresh loop
	•	Triangulation + UI render


These values match user-observed runs during EDTH Berlin.

⸻

6. MVP Boundaries

Included (MVP Completed)
	•	Real stereo DOA (GCC-PHAT)
	•	Real drone detection (ONNX)
	•	Real multi-node triangulation (LS)
	•	Persisted DB pipeline
	•	Time-window based multi-node fusion
	•	Live C2-style visualization

Out of Scope (Deferred)
	•	GNSS / PTP time sync
	•	RF/vision fusion
	•	Multi-target tracking
	•	Dynamic sound propagation modeling
	•	Adversarial robustness checks
	•	On-device model quantization (beyond Safari-safe ops)

⸻

7. Execution Map (File → Runtime Role)

WebApp
	•	app.legacy.js – control loop, inference, POST logic
	•	src/audio/bearing/* – all DSP: FFT, DOA, transforms
	•	index.html – runtime bootstrap, heading injection

Edge Adapter
	•	main.py – /webhook/edge + storage
	•	mappers.py – WirePacket mapping
	•	events.db – persistence

DSP Engine
	•	dsp_engine/* – triangulation + bearing alignment

C2 Integration
	•	c2_db_adapter.py – fetch → BearingSet
	•	c2_triangulation_adapter.py – BearingSet → results

UI
	•	streamlit_app.py – UI, KPIs, tracks

⸻

8. Known Limitations / Risks

Area	Impact	Notes
iOS Safari background throttling	Medium	Mitigated by coarse scheduling
No clock sync between nodes	High	Bearing alignment drift-sensitive
iPhone mic geometry variance	Medium	User heading param required
No SNR normalization	Medium	Future harmonic model needed
Single-target assumption	Medium	Fine for MVP; limits scaling


⸻

9. Recommended Next Steps (Helsing Perspective)
	1.	Replace ONNX MLP with lightweight CNN/Harmonic Encoder
	2.	Move FFT + DOA to WebWorker or WebGPU
	3.	Implement GNSS → PTP sync for multi-node deployments
	4.	Integrate RF envelope detector for spoofing suppression
	5.	Introduce α-β or EKF-based multi-track management
	6.	Hardening for low-SNR conditions (mean harmonic mask)

⸻

10. Conclusion

This MVP demonstrates:
	•	A fully functional passive acoustic ISR node
	•	Running entirely on iPhone Safari
	•	Producing real-time drone detections + validated bearings
	•	Feeding a working multi-node triangulation pipeline

While it is not production-ready, the core DSP and ML pipeline is real, correct, and aligned with Helsing’s tactical PM expectations.
