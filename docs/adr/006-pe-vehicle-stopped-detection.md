# ADR 006 — Parking Enforcement Vehicle Detection via Stopped-Car Signature

## Context

Parking enforcement officers arrive in marked vehicles before chalking tires. Detecting the vehicle's arrival gives earlier warning — potentially before the officer even leaves the car — and provides a second corroborating signal alongside the chalking detection.

The challenge: a stopped PE vehicle looks the same at the pixel level as any other parked car. Standard stationary masking (ADR 001) would suppress it. We needed a behavioral signature that distinguishes "car just stopped here" from "car has been parked here all along."

## Decision

Add a **stopped-car behavioral detector** (`PEVehicleAnalyzer`) with an **entry-velocity gate**:

1. When a `car` first appears in the zone, collect its center positions for `entry_frames` frames (default 5).
2. If the displacement over those frames is < `entry_min_px` (15 px), the car was already parked — mark it `SKIP` and never trigger on it.
3. If displacement ≥ `entry_min_px`, the car entered while moving — transition to `WATCHING`.
4. In `WATCHING`, count consecutive frames where per-frame velocity < `stop_px_per_frame` (3 px/frame). Once `sustained_frames` (30 frames ≈ 1 s at 30 fps) are accumulated, fire a VLM crop analysis.
5. VLM confirms enforcement markings (city emblem, "PARKING ENFORCEMENT" text, officer uniform near the vehicle).

`car` is exempt from the detector's stationary masking so that a stopped car continues to appear in zone detections.

## Consequences

**Good:**
- Early-warning signal — fires when the officer pulls up, ~10–30 s before chalking begins.
- Entry-velocity gate eliminates false positives from cars that were already on the street before the pipeline started.
- Reuses existing zone, VLM, notifier, and alert infrastructure with zero new HTTP endpoints.

**Watch out for:**
- Delivery trucks / taxis that stop briefly will enter the `WATCHING` phase; `sustained_frames` must be long enough (≥ 30 frames) to avoid false positives from short stops.
- `car` detection is noisy on YOLOv8n at low confidence thresholds — consider raising `INFERENCE_THRESHOLD` back toward 0.50 once the primary chalking detection is stable.
- The VLM prompt now covers three events (chalking, sweeper, PE vehicle); output token count increases slightly.
