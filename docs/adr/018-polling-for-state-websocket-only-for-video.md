# ADR 018 — Polling for application state; WebSocket only for the video feed

## Context

The dashboard needs four categories of live data from the server: pipeline stats, alert events, VLM pending jobs, and VLM debug-rejected frames. It also needs a continuous video feed. These have very different characteristics:

| Data | Update rate | Payload | Latency requirement |
|---|---|---|---|
| Stats | ~1/s (pipeline tick) | <1 kB JSON | Low (2 s lag acceptable) |
| Events | Rare (alert fires) | <2 kB JSON | Low (2 s lag acceptable) |
| Pending VLM jobs | ~0.5–2/s | <1 kB JSON | Low (1 s lag acceptable) |
| Debug rejected | ~0.5/s | ~5–20 kB (thumbnails) | Low (5 s lag acceptable) |
| Video feed | ~15 fps | ~20–80 kB JPEG/frame | High (must be near-real-time) |

## Decision

Use `setInterval` + `fetch` for the four JSON data sources. Use a WebSocket only for the video feed.

Each data source has its own hook:

- `useStats` — polls `/api/stats` every 2 s
- `useEvents` — polls `/api/events` every 2 s, deduplicates via a `lastEventTs` ref
- `usePending` — polls `/api/pending` every 1 s
- `useDebugRejected` — polls `/api/debug/rejected` every 5 s

`useVideoStream` opens `ws://host/ws/video`, receives binary JPEG blobs, creates object URLs, draws to a canvas via an `Image`, and revokes the URL after draw. On disconnect it waits 3 s and reconnects.

## Consequences

- Polling produces predictable server load at known intervals; a WebSocket for each data source would require server-side push logic and connection lifecycle management that adds complexity for no latency benefit at these update rates.
- Event deduplication on the client (`timestamp > lastEventTs`) means the server never needs to track per-client cursors — `/api/events` is a simple stateless snapshot of the last N events.
- The 5 s debug-rejected poll means a newly rejected frame appears in the drawer within 5 s, which is acceptable given this is a developer/diagnostic view.
- Video requires WebSocket because polling at 15 fps would produce 15 overlapping HTTP requests/second and the response ordering would be unpredictable. WebSocket preserves ordering and the binary blob can be sent directly without base64 encoding overhead.
