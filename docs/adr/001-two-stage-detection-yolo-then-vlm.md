# ADR 001 — Two-stage detection: YOLO pre-filter → VLM confirmation

**Status**: Accepted
**Date**: 2026-05-04

## Context

Parking enforcement events are rare. On a typical residential block the camera
runs for hours with nothing to detect. Two naive approaches both fail:

1. **Run VLM on every frame** — a 7B local model takes 1–2 s/frame; at 15 fps
   that's 15 VLM calls per second, impossibly slow. The Claude API version
   would cost ~$0.003/s ≈ $250/day for continuous analysis.

2. **Run only YOLO** — YOLOv8n classifies objects but cannot reason about *what
   the person is doing* (chalking) or identify sweeper-specific hardware
   (side brushes, water nozzles). It produces too many false positives.

Neither model alone solves the problem within the hardware and cost constraints
(8 GB VRAM, <$10/month API budget).

## Decision

Use a two-stage pipeline:

**Stage 1 — YOLO tracker (every frame, <30 ms)**
Detects `person`, `truck`, `motorcycle` within the street zone polygon.
Applies stationary masking and area filters to discard non-events.
Runs behavioral signature analysis (height decrease / velocity gate) as a
cheap in-process check.

**Stage 2 — VLM (only when signature fires, ~1–2 s)**
Receives either a tight crop (chalking) or the full frame (sweeper).
Returns the spec's structured JSON: `chalking_detected`, `sweeper_detected`,
`confidence`, `description`.
Acts as the authoritative classifier that suppresses YOLO false positives.

The behavioral signature gate reduces VLM calls from thousands per hour to
single digits, keeping both API costs and local inference load negligible.

## Consequences

**Positive:**
- Runs comfortably on a laptop CPU (YOLO stage) + 8 GB GPU (VLM stage)
- Claude API cost stays near zero on typical residential footage
- VLM adds explainability: `description` field is human-readable and logged
- Separate concerns: YOLO config controls sensitivity; VLM prompt controls accuracy

**Negative:**
- Two models to maintain and version-pin
- The signature gate introduces latency between the physical event and the VLM
  call (~0.5 s for the height-decrease window to fill) — acceptable for parking alerts
- If the behavioral signature is tuned too conservatively, real events are never
  escalated to the VLM (false negatives at stage 1 are invisible)
