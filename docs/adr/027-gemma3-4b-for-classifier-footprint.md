# ADR 027 — gemma3:4b as the VLM (footprint cut, offline-validated)

**Status:** Accepted
**Date:** 2026-05-24

## Context

`OLLAMA_MODEL=qwen2.5vl:7b` drives both the stage-1 chalking-detect call and the
person-type classifier (the classifier reuses the stage-1 analyzer —
`_classify_vlm = vlm`, [pipeline.py:283](../../src/pipeline.py#L283)). Loaded on
the GPU it occupies **~14 GB** (the "6 GB" in `ollama list` is disk; a vision
model loads its language weights + vision encoder + KV cache). The operator
asked whether a smaller model could lower that footprint on the shared 24 GB GPU
(also home to ollama's on-demand VLM and the host's LM Studio).

Per ADR-024's standing rule — **offline-validate model swaps on real
ground-truth before deploying** — candidates were scored with the existing
`POST /api/dataset/model-eval` harness over a stratified **199-event sample** of
human-labeled person types. (The chalking-*detect* task has only ~9 labels, so
it can't be evaluated; the person-type *classify* task has ~960 labels and is
the classifier's actual job — pre-filtering obvious non-chalkers to save VLM
calls.)

## Decision

Switch **`OLLAMA_MODEL` → `gemma3:4b`** (single model, both stages).

### Measured (199 labeled events, classify task)

| Model | Loaded VRAM | Accuracy | "unknown" rate |
|---|---|---|---|
| qwen2.5vl:7b (was) | ~14 GB | 34.2% | 9.0% |
| **gemma3:4b** | **~4.4 GB** | **35.2%** | **4.0%** |
| moondream | ~1.3 GB | 29.6% | 24.6% |

gemma3:4b **matches 7b on accuracy at ~⅓ the VRAM**, and emits fewer
`unknown`s (4% vs 9%) — i.e. it pre-filters *more*, the opposite of why ADR-024
rejected qwen-3b. moondream collapses to "everything is a pedestrian" (0/48
occupants, 0/42 deliveries, 24.6% unknown) — too weak.

The stage-1 *detect* path rides along on gemma3:4b **unvalidated** (no labels),
accepted because: detect barely fires (detection_rate 0.001) and the **wand gate
is the authoritative chalker detector** (ADR-024), so stage-1's exact behaviour
is low-stakes. Reversible via one env var if it misbehaves.

## Consequences

**Positive**
- Loaded VLM footprint **~14 GB → ~4.4 GB** with no classify-accuracy loss and
  better pre-filtering (lower unknown).
- More GPU headroom alongside ollama-confirm and the host LM Studio.

**Negative / watch out**
- **Detect stage unvalidated.** Mitigated by the wand gate; monitor stage-1
  positives after deploy and revert `OLLAMA_MODEL` if needed.
- **`.env`-only change** (not version-controlled) — captured here and in the
  inline `.env` comment so the rationale survives.
- `CONFIRM_OLLAMA_MODEL=qwen2.5vl:14b` is still unpulled (errors/no-ops, as
  noted in ADR-024) — untouched; the wand gate remains authoritative.
- Pulled-but-unused models remain on disk (`qwen2.5vl:32b` ~21 GB,
  `qwen2.5vl:3b`, `qwen2.5vl:7b`, `moondream`) — disk only, prune when desired.

## Principle reinforced (see also [L19](../LEARNINGS.md))

Across all three sizes accuracy sat at **~35%** — bigger did not help, smaller
(within reason) did not hurt. The 6-way taxonomy is the ceiling, not model
capacity, so the only real lever was footprint. This is ADR-024's lesson again,
now in the cost-cutting direction: **measure on the real workload — capacity you
can't use is just footprint.** It also surfaced a separate question for later:
at ~35% the classifier may not be earning its VLM cost vs leaning on the wand
gate.
