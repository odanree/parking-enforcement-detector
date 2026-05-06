# ADR 009 — Switch WebSocket Backend from websockets to wsproto

## Context

The dashboard streams annotated JPEG frames over a WebSocket at ~20 fps. Under load (mobile clients via ngrok, slow connections) two concurrent write paths existed on the same connection:

1. Our `send_bytes` loop in `video_stream`.
2. The `websockets` library's background keepalive ping task and close-frame acknowledgment handler.

Both paths call `write_frame → drain → _drain_helper`, which asserts `waiter is None or waiter.cancelled()`. The assertion fails when one write is already draining and a second starts. This produced `ERROR: keepalive ping failed` and `ERROR: data transfer failed` logs, and dropped mobile connections entirely on disconnect.

Setting `ws_ping_interval=None` suppressed the ping race but the close-frame race remained.

## Decision

Switch uvicorn's WebSocket implementation to `wsproto` by passing `ws="wsproto"` to `uvicorn.run()` in `src/main_web.py`. `wsproto` is a sans-I/O protocol library that does not share the legacy write-drain concurrency bug. The server is started via `python -m src.main_web` rather than bare `uvicorn` so this setting is always applied.

## Consequences

- `wsproto` must be installed (`pip install wsproto`), added to `requirements.txt`.
- The `websockets` package remains installed (used elsewhere in the ecosystem) but is no longer the active WS transport.
- Connection stability improves notably on high-latency paths (ngrok, mobile).
