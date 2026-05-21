# ADR 023 — Detection-accuracy rework: post-track gate, classifier pre-filter, pose priors, structured-evidence VLM

**Status:** Accepted
**Date:** 2026-05-21

## Context

A dataset audit (22,921 events in ChromaDB) surfaced four compounding problems:

1. **Stage-1 is flooded with noise.** ByteTrack runs at a low-confidence
   threshold (0.26) so it can re-associate partially occluded tracks, but the
   returned detections were never re-filtered against the user-configured
   inference threshold (0.65). YOLO mean confidence on stored events: 0.29.
   68 % of chalking events were the lenient VLM rejecting shadows.
2. **Confidence is uncalibrated across backends.** Qwen2.5VL returned
   confidence on a 0–100 scale; 10 % of records had `confidence > 1.5`,
   silently breaking the `>= 0.80` cooldown gate and every kanban display.
3. **RAG retrieval is structurally degenerate.** 78 unique descriptions
   across 22,921 events — text embeddings collapse to clusters of identical
   points. `zone_pedestrian` events all shared one hard-coded string.
4. **The chalking prompt has accreted into a 4 KB B1/B2/B3 decision tree
   (ADRs 011, 017, 020)** that Haiku-class models cannot apply reliably.

`classify_person()` (six-way classifier: pedestrian / occupant /
worker_landscape / worker_delivery / chalker / unknown) existed in the
analyzer but was never called by the pipeline. `person_type` was 0 %
populated despite the dataset having a usable signal for stratification.

## Decision

A single rework branch lands six coupled changes:

1. **Post-track confidence gate** ([object_detector.py][1]). After ByteTrack
   association, drop detections whose confidence is below the configured
   `INFERENCE_THRESHOLD`. The low-thresh re-association behaviour is
   preserved inside the tracker; only the *returned* detections are gated.

2. **VLM confidence normalisation** ([analyzer.py][2]). `_normalize_confidence()`
   maps values > 1.5 to `/100` (Qwen-style 0–100), clamps anything out of
   range, and is applied in both `_normalize` and `_normalize_classify`.

3. **`classify_person()` wired as a per-track cached pre-filter**
   ([pipeline.py][3]). On the first chalking-eligible frame for each track,
   the analyzer's six-way classifier runs once. The result is cached for the
   life of the track and recorded in vector-store metadata. Classes listed
   in `PERSON_CLASSIFIER_SKIP` (default: `pedestrian,occupant,worker_delivery`)
   short-circuit the chalking VLM call entirely.

4. **Diversified zone-pedestrian descriptions**. Encode 3×3 spatial grid,
   bbox-size bucket, YOLO confidence, and the classifier label into the
   description string so sentence embeddings produce real spread instead of
   a single point.

5. **Pose estimation as deterministic prior** ([pose_estimator.py][4]).
   Opt-in YOLOv8m-pose stage computes `is_crouching` (knee angle +
   hip-below-knee), `hand_low` (wrist in bottom 25 % of bbox),
   `wrist_near_vehicle` (within `POSE_CONTACT_PX` of a vehicle bbox).
   The flags are prepended to the user prompt as `PRIOR SIGNALS` and
   carried through to the pipeline trace and dataset metadata.

6. **Structured-evidence VLM prompt + Python policy layer**
   ([apply_chalking_policy][5]). Opt-in via `VLM_STRUCTURED_PROMPT`. The
   VLM returns observation flags only; Python combines them via documented
   precedence (trunk-open → false; tool-visible → true; crouch+close →
   true; rear-to-front+close → true with capped confidence). The 4 KB
   legacy prompt remains the default for backward compatibility.

A new CLI (`scripts/dataset_maintenance.py`) provides:
`stats`, `purge-vlm-errors`, `purge-shadow`, `fix-confidence`,
`backfill-person-type`, `phash-cluster`, `rewrite-zoneped`, `clip-backfill`.
All destructive subcommands default to dry-run.

## Consequences

**Positive**
- Stage-1 VLM call rate drops ~10× by gating sub-0.65 person detections
  before they reach the analyzer.
- `confidence` is now reliable across backends; downstream thresholds work as
  intended on Ollama deployments.
- `person_type` becomes a first-class signal: events are stratifiable by
  pedestrian / occupant / worker / chalker without manual labelling.
- The legacy prompt and the structured-evidence prompt can be A/B compared
  against the same footage by toggling one env var.
- Pose signals give the system a deterministic alternative to "model
  guesses crouch from a 50-pixel bbox" — works regardless of camera
  distance.

**Negative / watch out**
- `classify_person()` adds one VLM call per new track. Cost amortises
  because subsequent chalking-VLM calls on that track may be skipped, but
  high-churn track footage (poor tracker tuning, frequent re-IDs) will
  multiply classifier calls. Tune `_CLASSIFY_ENABLED` per camera.
- The post-track gate at 0.65 may suppress real detections of small/distant
  persons. Lower `INFERENCE_THRESHOLD` per camera if recall drops.
- The policy layer makes the chalking criterion explicit. This is the
  right trade-off but every change to the rules now requires code, not a
  prompt edit. ADR 011 / ADR 017 / ADR 020 remain valid for the legacy
  prompt path.
- Open-CLIP is added as an optional dependency; ~1 GB on first install.
  CLIP is opt-in via `CLIP_EMBEDDINGS_ENABLED`.

## Migration

Existing deployments can adopt this incrementally:

```bash
# 1. Inspect the dataset.
python -m scripts.dataset_maintenance stats

# 2. Purge accumulated noise.
python -m scripts.dataset_maintenance purge-vlm-errors --apply
python -m scripts.dataset_maintenance purge-shadow --apply
python -m scripts.dataset_maintenance fix-confidence --apply
python -m scripts.dataset_maintenance rewrite-zoneped --apply

# 3. (optional) Enable classifier and pose.
export PERSON_CLASSIFIER_ENABLED=true
export POSE_ESTIMATION_ENABLED=true
# pip install open-clip-torch
export CLIP_EMBEDDINGS_ENABLED=true
python -m scripts.dataset_maintenance clip-backfill --apply --limit 5000
python -m scripts.dataset_maintenance backfill-person-type --apply --limit 500

# 4. (optional) Try the structured-evidence VLM mode.
export VLM_STRUCTURED_PROMPT=true
```

[1]: ../../src/detection/object_detector.py
[2]: ../../src/vlm/analyzer.py
[3]: ../../src/pipeline.py
[4]: ../../src/detection/pose_estimator.py
[5]: ../../src/vlm/analyzer.py
