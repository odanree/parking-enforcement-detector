# ADR 004 — Live dashboard via WebSocket MJPEG stream

**Status**: Accepted
**Date**: 2026-05-04

## Context

The pipeline runs as a background process with no visibility into what the
camera sees, which detections are firing, or whether the zone is positioned
correctly. Debugging zone placement requires restarting the process after each
YAML edit — a slow loop that makes calibration frustrating.

Two approaches were considered:

1. **Server-sent MJPEG stream** (`multipart/x-mixed-replace`) — classic IP
   camera approach, works in `<img>` tags, but no bidirectional channel for
   future control commands.
2. **WebSocket binary stream** — send JPEG frames as binary messages, render
   on a `<canvas>` element — bidirectional, same connection can carry control
   messages in both directions if needed.

## Decision

Use a WebSocket endpoint (`/ws/video`) that sends JPEG-encoded annotated frames
as binary messages at ~15 fps. The client renders them on a `<canvas>` using
`URL.createObjectURL` → `Image` → `drawImage`, which avoids base64 overhead
and is smooth on all modern browsers.

Frames are annotated server-side in `pipeline.py` before being pushed to
`AppState`. The annotation draws:
- Zone polygon (cyan, dashed when inactive)
- Bounding boxes per detection class
- Track ID labels for detections inside the zone
- Red boxes for tracks that triggered an active alert

The pipeline pushes at most every N frames (capped at 15 fps) so annotation
and encoding don't dominate CPU time.

## Consequences

**Positive:**
- Real-time visual confirmation that detection and zone are working
- Bounding boxes + track IDs make it easy to spot false positives
- WebSocket is bidirectional — future control commands (PTZ, manual alert) can
  reuse the same infrastructure
- Auto-reconnects client-side on disconnect

**Negative:**
- Annotating and JPEG-encoding every displayed frame adds ~5 ms per frame of
  CPU overhead on the pipeline thread
- Multiple browser tabs each hold a WebSocket connection and receive duplicate
  frame data — not a concern for single-user home deployment
- Canvas-based display doesn't support native browser video controls (seek,
  download) — intentional for a live feed
