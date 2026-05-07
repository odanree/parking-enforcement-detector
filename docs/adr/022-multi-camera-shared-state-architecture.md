# ADR 022 — Multi-camera architecture: per-camera AppState with merged API responses

## Context

A second camera covering a different street angle was added. The options for supporting two simultaneous video streams were:

**Option A — Single pipeline, multiplexed stream**: run one pipeline thread that reads from two streams alternately. Simple but impractical: frame timing from two sources cannot be interleaved cleanly, and zone/privacy/playback state would need to be split internally anyway.

**Option B — Two independent pipeline processes**: separate OS processes with IPC. Provides isolation but complicates shared config, adds overhead, and makes the web layer more complex (two servers or a proxy).

**Option C — Two pipeline threads sharing one web process**: each camera gets its own `AppState` instance and a dedicated daemon thread running `pipeline.run()`. A single FastAPI process owns all `AppState` instances and merges their outputs at the API layer.

## Decision

Option C. Two `AppState` instances in a list (`states[0]`, `states[1]`) are created at app startup. Each pipeline thread is passed its own state object and a camera-specific zone key:

```python
states = [AppState(0), AppState(1)]
threading.Thread(target=pipeline.run, args=(states[0],), kwargs={"zone_key": "street_zone"})
threading.Thread(target=pipeline.run, args=(states[1],), kwargs={"zone_key": "street_zone_1", "video_path": ...})
```

`AppState` is unchanged — it has no knowledge of other cameras. All per-camera isolation is implicit in the separate instances.

**API merging strategy:**
- `GET /api/events` — merges both cameras, injects `"camera": cam_id`, sorts by timestamp descending.
- `GET /api/pending` — same merge pattern.
- `GET /api/debug/rejected` — same merge pattern.
- `GET /api/stats` — returns cam-0 state (pipeline running, paused, fps, playback controls) with `total_chalking` and `total_sweeper` summed across both cameras.
- `POST /api/pipeline/pause`, playback controls — cam-0 only; each camera would need its own endpoint if independent playback control is required.

**Per-camera resources:**
- Zone polygon: `street_zone` and `street_zone_1` in `detection.yaml`, updated independently via `POST /api/zone/{cam_id}`.
- Privacy regions: `config/privacy_0.json` and `config/privacy_1.json`, updated independently via `POST /api/privacy/regions/{cam_id}`.
- WebSocket stream: `/ws/video/0` and `/ws/video/1`.

**Frontend:** event cards show a "Cam 1" badge when `ev.camera > 0`. Cam 0 events show no badge (default, most common camera).

## Consequences

- **No lock contention between cameras**: each `AppState` has its own `threading.Lock()`. The two pipeline threads write to separate state objects and never block each other.
- **Merge is O(n) at read time**: sorting merged event lists on every `/api/events` poll is negligible for the deque sizes in use (≤50 events per camera).
- **Playback controls are cam-0 only**: the `state` alias points to `states[0]`. A second video file on camera 1 can be started/paused/seeked independently only by adding cam-1-specific endpoints. Not yet implemented.
- **VLM thread pool is per-pipeline**: each camera has its own `ThreadPoolExecutor(max_workers=2)`. Under sustained detection load on both cameras, up to 4 concurrent VLM calls are possible. This is acceptable at Claude Haiku latency and pricing but would saturate a local Ollama instance.
- **`DELETE /api/debug/rejected` clears all cameras**: intentional — the debug drawer shows a merged view and clearing it is a single action.
