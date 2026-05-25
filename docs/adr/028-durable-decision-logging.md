# ADR 028 — Durable per-decision logging

**Status:** Accepted
**Date:** 2026-05-24

## Context

A delivery driver showed up on the kanban as `X Worker → Rejected 100%`, and the
question "did the VLM classify this as delivery, or did similarity search?"
could not be answered from the running system:

- The per-stage `pipeline_trace` (classify / first-pass / RAG / confirm) is held
  in an **in-memory ring buffer** that rotates; `GET /api/pipeline/trace`
  returned 0 once it had turned over.
- The `pipeline_trace` is persisted on the event record **only for alerts and
  rejected-VLM records** — and recent live events weren't in the vector store at
  all (dataset latest was days old). **Classifier short-circuit skips**
  (`pedestrian`/`occupant`/`delivery`/… → never reach the chalking VLM) are the
  highest-volume decisions and were **never written anywhere**.
- The only existing log for a classify decision was
  `logger.debug("classify_person failed")` — i.e. nothing on the success path.

So the most common decision the pipeline makes (classify → skip) left no durable
trace, and "why was track X classified/rejected?" was unanswerable after minutes.

## Decision

Emit **one structured `INFO` line per VLM decision**, at the single choke point
in `run()` where completed VLM jobs are harvested (`fut.result()` —
[pipeline.py](../../src/pipeline.py)), where `track_id`, `camera_id`, the full
`pipeline_trace`, and the final verdict are all in scope:

```
DECISION track=13308 cam=0 kind=chalking classify=delivery classify_conf=0.82 \
         model=gemma3:4b detected=False conf=1.00 outcome=skip
```

- **Format: `key=value`** — human-skimmable and greppable
  (`grep "track=13308" logs/detector.log` gives the trail). JSON was rejected as
  less skimmable for the ad-hoc "why did X happen" lookups this is for.
- `outcome` ∈ {`skip` (classifier short-circuit), `alert`, `reject`} disambiguates
  a pre-filter skip from a real VLM "no".
- Single choke point rather than per-`return` inside `_two_stage` (4+ exits) —
  the result dict carries everything needed.

Companion change: `detector.log` switched from a plain `FileHandler` to a
**`RotatingFileHandler` (10 MB × 5)** — it was already unbounded at 23 MB, and
per-decision lines add volume.

## Consequences

**Positive**
- "Why was this event classified/rejected" is answerable from
  `logs/detector.log` long after the in-memory trace rotates — including the
  short-circuit skips that were previously invisible.
- Confirmed the live path in practice: the delivery case was a **classifier
  (gemma3:4b) short-circuit**, not similarity search (RAG runs only post-classify
  and never assigns `person_type`).
- The log can no longer grow without bound.

**Negative / watch out**
- One log line per classified track — bounded by event volume (rare) and capped
  by rotation; if volume ever spikes, lower `backupCount` or raise `maxBytes`.
- Logs are **not** a queryable store. If decision analytics are ever needed at
  scale, persist a compact skip record to the vector store or a separate table —
  deliberately avoided here to not bloat the labeled dataset with every skipped
  pedestrian.
- The `DECISION` line reports the classifier/first-pass `backend`; on the
  short-circuit path that is the classify model (now `gemma3:4b`, [ADR-027](027-gemma3-4b-for-classifier-footprint.md)).

## Principle

**A decision the system makes silently is a decision you can't debug.** The
highest-volume path (classify → skip) had no durable record precisely because it
short-circuits early; ephemeral in-memory traces are not an audit trail. Log the
outcome at the choke point, in a format you can grep under pressure.
