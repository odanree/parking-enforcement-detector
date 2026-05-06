# ADR 007 — VLM Calls Run in a Background ThreadPoolExecutor

## Context

The VLM analysis step (Ollama llava:7b or Claude Haiku) can take 5–60 seconds per call. When called synchronously inside the main pipeline loop it blocked frame processing entirely — the WebSocket stream froze and bounding boxes stopped updating for the full duration of the inference.

## Decision

Dispatch VLM calls via a `ThreadPoolExecutor(max_workers=2)`. Each trigger (chalking sample, sweeper match, PE vehicle stop) submits a `Future` and immediately returns control to the loop. A harvest block at the top of each iteration checks `fut.done()` and fires the alert when the result is ready.

In-flight jobs are tracked in a `_vlm_jobs: dict[tuple[str, int], Future]` keyed by `(kind, track_id)`. A second submission for the same key is skipped until the first resolves.

The snapshot frame is captured with `frame.copy()` at submission time so the correct frame is sent to the notifier even if the result arrives 60 frames later. Privacy redaction is applied at harvest time (current privacy state, not submission-time state).

## Consequences

- Pipeline loop runs uninterrupted at full frame rate regardless of VLM latency.
- Alert delivery is delayed by VLM inference time (expected, not a regression).
- Two concurrent VLM calls can run in parallel (one chalking, one sweeper), which is fine for Ollama since it queues internally.
- If a tracked object leaves the zone while its VLM call is in flight, the alert still fires when the future resolves — acceptable because the snapshot was valid at submission time.
