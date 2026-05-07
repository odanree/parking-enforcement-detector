# ADR 020 — Multi-frame VLM context window for chalking detection

## Context

The single-frame VLM crop approach had two failure modes that a static image cannot resolve:

1. **Open trunk not in crop** — the chalking crop is centered tightly on the person. A vehicle with an open trunk a metre away sits outside the crop boundary. The VLM has no evidence to apply the trunk-gate exclusion, so it falls through to evaluating posture and can produce a false positive.

2. **Ambiguous mid-action frames** — a single frame may catch the person mid-stride between positions. A PE officer approaching a rear wheel looks identical to a car owner walking past. Temporal context (person dwells at wheel across multiple frames vs. briefly passes) is the discriminating signal.

## Decision

Send two types of images to the VLM per evaluation:

- **Image 1 — scene crop** (`_crop_scene_bytes`): wide field of view (10× bbox width, 4× bbox height) centred on the tracked person. Purpose: vehicle-state assessment (open trunks, doors, hatches). Not used for posture or tool detection.
- **Images 2–N — detail crops** (`_crop_wide_bytes`): tighter crops sampled from a rolling frame buffer (configurable `frame_buffer_size` and `buffer_sample_every_n`). Purpose: posture, tool, and dwell-time assessment across the evaluation window.

The VLM prompt explicitly assigns roles: "Apply STEP 1 (trunk/door gate) using the scene image. Apply STEP 2 across all detail frames in totality."

The `ChalkingAnalyzer` maintains a per-track `deque` of JPEG bytes sampled every `buffer_sample_every_n` frames. `entry_frames` is set to `(frame_buffer_size - 1) × buffer_sample_every_n + 1` so the buffer is full on the first VLM call.

Tuned values for 15 fps video: `frame_buffer_size: 3`, `buffer_sample_every_n: 4` → ~0.5 s window per evaluation.

## Consequences

- **Reduced false negatives from trunk-gate miss**: the scene crop is wide enough to show an adjacent open trunk even when the tight crop would not.
- **Temporal discrimination**: CONDITION B (sustained presence at rear wheel) can now be assessed across frames instead of inferring it from a single posture.
- **Increased token cost per call**: sending 4 images vs. 1 increases cost roughly 3×. At Claude Haiku pricing (~$0.0001/call single-frame), this remains negligible for the detection frequency used.
- **`entry_frames` must be recalculated** when `frame_buffer_size` or `buffer_sample_every_n` changes. Setting `entry_frames` too high causes subjects to leave the zone before any VLM call fires; setting it too low causes the first call to have fewer frames than expected.
