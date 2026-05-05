# ADR 005 — Live zone editor in dashboard with hot-reload (no process restart)

**Status**: Accepted
**Date**: 2026-05-04

## Context

Zone calibration is iterative: place the polygon over the curb area, watch the
live feed to see if detections fire in the right place, adjust, repeat. Two
problems with the initial YAML-only approach:

1. **No visual editor** — the user must manually guess pixel coordinates in
   `config/detection.yaml`, restart, and look at the feed to see where the box
   landed. Non-technical users (or anyone calibrating at night under camera
   distortion) cannot do this reliably.

2. **Slow feedback loop** — each YAML edit requires restarting uvicorn (~15 s:
   uvicorn restart + YOLO model load + RTSP stream reconnect). At 5–10 iterations
   to dial in a polygon, that's 2+ minutes of dead time.

## Decision

Build an interactive polygon editor directly into the dashboard video feed, and
make zone changes take effect without restarting the process.

**Zone editor UI** (canvas overlay in the browser):
- An "Edit Zone" button switches the video overlay canvas to interactive mode
- Click on the live video to add polygon vertices; drag to reposition; right-click to remove
- The polygon is drawn on top of the live annotated feed in real time so the
  user can see exactly which part of the frame the zone covers
- Save / Cancel buttons: Save posts the polygon to `POST /api/zone`, Cancel reverts

**Hot-reload mechanism** (shared `AppState`):
- `AppState` holds the canonical zone polygon at runtime and a monotonic version counter
- `POST /api/zone` updates `state.zone_polygon`, increments `_zone_version`, and
  persists the polygon to `config/detection.yaml` for restart durability
- The pipeline checks `state.get_zone_version()` each frame (one lock + integer
  read, ~1 μs); when the version changes it rebuilds `ZoneFilter` in place —
  the RTSP stream and YOLO model are untouched

This was preferred over a file-watch approach (which adds per-frame I/O and a
partial-write race condition) and over a separate CLI calibration tool (which
wouldn't show the live feed).

## Consequences

**Positive:**
- Full calibration loop (draw polygon → see effect on live feed) takes ~1 s vs ~15 s
- Non-technical users can calibrate without touching YAML or restarting anything
- Zone takes effect within one frame (~66 ms at 15 fps) after clicking Save
- YOLO model stays loaded and RTSP stream stays open during calibration
- Persisted to YAML automatically — zone survives process restarts

**Negative:**
- Zone editor only available when running the web dashboard (`uvicorn`);
  headless mode (`python -m src.main`) reads zone from YAML at startup only
- If the process crashes between the API updating `AppState` and the YAML write
  completing, the runtime zone and the file can diverge for that session
  (acceptable risk for a home-use tool; a write-then-swap pattern would fix it)
- Canvas polygon editor does not support snapping or grid alignment — precise
  pixel-boundary zones require manual YAML editing after a rough UI calibration
