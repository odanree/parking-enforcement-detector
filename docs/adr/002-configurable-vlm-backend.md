# ADR 002 — Configurable VLM backend (Claude API vs. Ollama)

**Status**: Accepted
**Date**: 2026-05-04

## Context

Two VLM deployment models exist with different trade-off profiles:

| | Claude API (Haiku) | Ollama (local LLaVA 7B) |
|---|---|---|
| Setup | API key only | 8 GB VRAM, model download |
| Cost | ~$0.0001/call | $0 per call |
| Latency | 1–3 s (network) | 1–2 s (local GPU) |
| Accuracy | Higher (larger model) | Adequate for structured JSON |
| Privacy | Frames sent to Anthropic | Frames stay on device |
| Availability | Requires internet | Works offline |

Early development naturally starts with Claude (no GPU needed), but a home
deployment likely prefers Ollama (cost and privacy). There is no single correct
answer — it depends on hardware, budget, and privacy requirements.

## Decision

Abstract the VLM behind a single `VLMAnalyzer` class with a `backend` parameter
(`"claude"` | `"ollama"`). The backend is selected at runtime via `VLM_BACKEND`
environment variable. Both backends receive the same prompt text and return the
same dict schema, so no other component knows which backend is active.

Claude backend uses `claude-haiku-4-5-20251001` with prompt caching on the
system prompt (cache TTL 5 min, refreshed each hit — ~75 % token reduction on
repeated calls).

Ollama backend targets `llava:7b-v1.6-mistral-q4_K_M` by default (4-bit
quantised, fits in 6 GB VRAM with headroom).

## Consequences

**Positive:**
- Switch backends with one env var change — no code deploy
- Development cycle uses Claude (fast iteration, no GPU required)
- Production deployment can use Ollama (offline, zero marginal cost)
- Prompt caching makes the Claude path cost-efficient even for active monitoring

**Negative:**
- Must test both backends separately — a prompt that works well on Claude may
  produce malformed JSON on a local model (see L3 in LEARNINGS.md)
- Ollama response quality varies significantly between quantisation levels;
  INT8 is noticeably more accurate than INT4 on the chalking task
- The `_parse_json` helper must be robust to both backends' output quirks
