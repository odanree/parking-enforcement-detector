# ADR 008 — Chalking Detection via Periodic VLM Sampling

## Context

The original chalking trigger used a height-decrease heuristic: if a tracked person's bounding box shrank by a threshold percentage, it inferred a crouching motion (bending to chalk a tire). At typical camera distances (10–20 m), persons occupy only 30–50 px tall. Bounding box jitter from the detector exceeded the crouch signal, producing both missed detections and false positives.

## Decision

Replace the height-decrease heuristic with periodic VLM sampling:

- After a person has been tracked for `entry_frames` (10) frames, send a wide-context crop to the VLM every `sample_every_n` (30) frames (~1 s at 30 fps).
- The crop is padded 3× bbox width horizontally and 2× bbox height vertically so the model sees the full person, any held tool, and adjacent vehicles.
- A `cooldown_seconds` (60 s) gate prevents re-alerting the same track_id within the cooldown window.
- The VLM prompt explicitly notes the distant-camera context: "person may appear upright — focus on any extended object toward a wheel."

## Consequences

- Detection is no longer gated on any geometric heuristic, so it works at any camera distance and for any chalking posture.
- VLM call rate is bounded: one call per tracked person per second (further throttled by the cooldown after a positive).
- False negative window: up to `sample_every_n` frames between when chalking starts and when the VLM is queried. Acceptable — chalking typically takes 5–15 s per tire.
- VLM cost increases slightly vs. the heuristic approach, mitigated by the cooldown gate and async execution (ADR 007).
