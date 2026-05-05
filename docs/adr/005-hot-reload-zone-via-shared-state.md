# ADR 005 — Zone hot-reload via shared AppState (no process restart)

**Status**: Accepted
**Date**: 2026-05-04

## Context

Zone calibration is iterative: place the polygon, watch the feed, adjust,
repeat. If each adjustment requires restarting the uvicorn process the feedback
loop takes ~15 s per iteration (restart + YOLO model load + stream reconnect).

Two options to make the zone update without a restart:

1. **File watch** — pipeline polls `config/detection.yaml` for mtime changes,
   reloads when the file changes. Simple, but adds I/O on every frame and
   creates a race condition if the file is partially written.
2. **Shared in-process state** — `AppState` holds the canonical zone polygon.
   The API endpoint updates `state.zone_polygon` and increments a version
   counter. The pipeline compares the version counter each frame and rebuilds
   `ZoneFilter` only when it changes.

## Decision

Use the version-counter approach via `AppState`. The pipeline checks
`state.get_zone_version()` each frame (a single lock + integer read, ~1 μs)
and reconstructs `ZoneFilter` only when the counter changes. The API endpoint
also persists the new polygon to `config/detection.yaml` so the updated zone
survives a process restart.

The zone editor UI in the dashboard posts to `POST /api/zone`, which updates
state and saves the file atomically from the FastAPI thread — no file-watch
race condition.

## Consequences

**Positive:**
- Zone takes effect within one frame (~66 ms) after clicking Save
- No process restart means YOLO model stays loaded and the RTSP stream stays
  open — zero interruption to monitoring during calibration
- Single source of truth: `AppState.zone_polygon` is authoritative at runtime;
  `config/detection.yaml` is authoritative at startup
- Version counter is cheap: one mutex acquisition per frame regardless of zone
  size or polygon complexity

**Negative:**
- If the process crashes between the API updating `AppState` and the file write
  completing, the in-memory zone and the YAML file can diverge for that one
  frame — the YAML write uses Python's built-in buffering so partial writes are
  possible in theory (acceptable risk for a home-use tool)
- Headless (`python -m src.main`) mode reads the zone only from YAML at
  startup; live zone editing requires the web dashboard
