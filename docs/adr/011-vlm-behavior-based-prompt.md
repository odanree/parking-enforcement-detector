# ADR 011 — VLM Prompt: Behavior-Based Chalking Detection

## Context

The original chalking prompt required the VLM to see a visible tool: "holding or extending a stick, chalk, or long tool toward a wheel." From a camera mounted 5–10 m overhead, a chalk stick held close to the ground is typically 2–4 px wide — below the threshold of reliable detection for any 7B-class VLM. The debug drawer (ADR 010) confirmed that the VLM was returning high-confidence negatives on frames where a person was clearly adjacent to a vehicle's wheel with no visible tool present.

## Decision

Rewrite the chalking detection criterion from tool-visibility to behavioral indicators:

**Old:** "Is a person holding or extending a stick, chalk, or long tool toward the ground, a wheel, or tire area?"

**New:** Flag TRUE if ANY of:
- Person is crouching, bending, or leaning toward a wheel/tire area
- Person is making close physical contact with a wheel or tire
- Person is holding or extending any object toward a wheel or the ground near a tire
- Person is standing very close to a parked vehicle's rear wheel for more than a moment

The system prompt was also updated to explicitly tell the model: "When in doubt about chalking, return true with a lower confidence score rather than false — missed detections are worse than false positives here."

## Consequences

- False positive rate increases: a person tying their shoe near a car may now be flagged.
- False negative rate decreases: the actual chalking behavior (person close to rear wheel) is flagged correctly.
- The VLM confidence score now carries more signal — a 30% confidence positive is a genuine ambiguous case worth human review, vs. before when 100% confidence negatives were common even on clear chalking frames.
- This trade-off is correct for the use case: missed chalking events are the primary failure mode; false positives are reviewed by a human and dismissed.
