# ADR 024 — Lean resource profile and VLM model selection

**Status:** Accepted (CPU-only torch decision superseded by [ADR-026](026-gpu-inference-for-yolo.md))
**Date:** 2026-05-23

## Context

The detector runs on a shared Windows workstation (RTX 3090, 24 GB) alongside
the operator's desktop apps (browsers, Slack, and especially LM Studio, which
parks a ~13 GB LLM in VRAM). Docker runs in WSL2, so all containers share one
`VmmemWSL` memory pool. Measured idle footprint of the parking pipeline:

- **ollama ~5.8 GB RAM** — the qwen2.5vl weights resident/mmap-cached even when
  no model was actively loaded (`ollama ps` empty).
- **detector ~2.3 GB RAM + ~8 CPU cores** — YOLO + YOLOv8m-pose + CLIP + MOG2
  running per-frame on **CPU-only torch** (the Dockerfile installs the CPU
  build deliberately to keep the image small), across 2 cameras at ~18 fps.

GPU headroom is dictated by host apps, not containers — only ollama uses the
GPU, on demand. The 14 B confirm model referenced in `.env`
(`CONFIRM_OLLAMA_MODEL=qwen2.5vl:14b`) was never even pulled, so the confirm
stage had been erroring (`model not found` / timeouts) — the source of the
`VLM HTTP 500` / `timeout` rows seen in the dataset.

Two questions: how to shrink the footprint, and whether a different VLM (bigger
for accuracy, or smaller for footprint) is worth adopting.

## Decision

### Lean resource profile (applied)

- **`OLLAMA_KEEP_ALIVE=30s`** (ollama service env) — the VLM fires only when a
  person is near a vehicle (rare), so unload the model 30 s after the last call.
  Idle ollama dropped **5.8 GB → 91 MB**. Cost: ~3-5 s cold-load on the first
  event after idle (acceptable for parking).
- **`POSE_ESTIMATION_ENABLED=false`** — pose ran a second model per frame and
  was a weak prior at this camera distance (keypoints unreliable on 20-90 px
  people). Biggest CPU saver; detector RAM 2.3 GB → ~0.9 GB.
- **`CLIP_EMBEDDINGS_ENABLED=false`** — frees ~350 MB; no functional loss since
  no UI consumes image-similarity yet. The `chalking_evals_clipv2` index is
  preserved for when it's wired up.
- **Kept**: the **wand gate** (`WAND_GATE=promote`, the actual fisheye-chalking
  detector) and the **person classifier** (`PERSON_CLASSIFIER_ENABLED=true`).

Result: pipeline idle footprint **~8 GB → ~1 GB**, spiking only during events.
The detector still pegs ~8-9 CPU cores; the only meaningful further cut is a
`DETECT_EVERY_N` frame-skip, deliberately **not** adopted (operator wants every
frame for now).

### VLM model selection (qwen2.5vl:7b retained)

Keep **qwen2.5vl:7b** as the primary/first-pass + classifier model. Three
alternatives were evaluated and rejected, each by **offline validation on real
snapshots before any deploy**:

| Candidate | For | Verdict |
|---|---|---|
| 14 B classify | better delivery-vs-pedestrian | **No** — even 32 B was *more* confident the test subject was a pedestrian; the delivery cue simply wasn't in the frame. Bigger model can't add information that isn't there. (14 B also wasn't pulled.) |
| Smaller confirm (vs 14 B) | confirm stage that fits VRAM | Moot — 14 B confirm never fit / wasn't pulled; with the wand gate as the real detector, the confirm stage matters far less. |
| 3 B primary | smaller footprint | **No** — markedly weaker: defaulted to low-confidence `unknown`, which *reduces* pre-filtering (`unknown` isn't skipped) → *more* per-event compute, not less. And `keep_alive` already solved the idle-RAM problem 3 B would have addressed. |

## Consequences

**Positive**
- Idle footprint cut ~8×; VRAM/RAM are only consumed during actual detection.
- Avoided a 9 GB model pull (14 B) and a model swap (3 B) that looked good on
  paper but degraded behaviour — caught by offline tests, not in production.
- Classification quality preserved (7 B), so the skip-set pre-filter keeps
  working.

**Negative / watch out**
- Pose and CLIP are off, so their signals/features are unavailable until
  re-enabled (both are opt-in env flags; flip back when needed).
- Detector CPU (~8-9 cores) is unchanged — frame-skip is the lever if it ever
  becomes a problem.
- `keep_alive=30s` adds cold-load latency to the first event after a quiet
  period; fine for parking, would not be for high-frequency triggering.
- The `.env` still references `qwen2.5vl:14b` for confirm, which isn't pulled —
  confirm calls will error/no-op. Acceptable because the wand-gate promote path
  is the authoritative detector; revisit if the confirm stage is reinstated.

## Principle established

**Offline-validate model swaps on real ground-truth snapshots before
deploying.** This ADR's three rejections, plus the earlier decision to keep the
classifier on 7 B, all came from one-shot offline comparisons that cost minutes
and prevented wasted pulls/rebuilds and regressions. Bigger isn't automatically
better (it can't supply missing visual cues); smaller isn't automatically
cheaper (weaker classification can increase downstream compute). Measure on the
actual data first.
