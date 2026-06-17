# ADR 029 — Langfuse observability for the Claude VLM path

**Status:** Accepted
**Date:** 2026-06-16

## Context

[ADR-028](028-durable-decision-logging.md) made every VLM decision durably
greppable via one `DECISION key=value` line per track. That answers *what* the
system decided, not *why* a given Claude call produced the JSON it did.

Specifically, when a kanban event looks wrong and the model in scope was Claude
(`VLM_BACKEND=claude`, `CONFIRM_BACKEND=claude`, or `REEVAL_BACKEND=claude`):

- The `_usage` block we capture in `_analyze_claude` carries token counts
  forward into the policy result, but it's not persisted with the original
  prompt + image + raw response. Reproducing the call means re-running the
  image through `curl` against the API — which loses `cache_control` state,
  truncates the prior conversation, and doesn't preserve the multi-frame
  ordering the strict prompt depends on.
- Adjacent: a portfolio-wide Anthropic API key reorg moved this project onto
  a dedicated `personal-dev` key for spend attribution. Console aggregation
  by key tells us *how much* the detector spends, not *which kind* of call
  (classify vs. analyze vs. structured re-eval vs. multi-frame confirm).
- ADR-011's behavior-based prompt and ADR-023's accuracy rework both rewrite
  prompts iteratively. Without a trace store we can't diff a prompt change's
  effect on real images without re-running the dataset eval every time.

## Decision

Wrap the two public `VLMAnalyzer` entry points with Langfuse's `@observe()`
and attach Claude's model + token usage from inside the two backend call
sites via `langfuse_context.update_current_observation()`:

```python
@observe(name="vlm.analyze", capture_input=False)
def analyze(self, image_bytes, kind="", prior_signals=None):
    langfuse_context.update_current_trace(
        name="vlm.analyze",
        metadata={"backend": self._backend, "model": self.model_name,
                  "kind": kind, "structured": self._structured,
                  "frames": frames_count, "prior_signals": prior_signals},
    )
    ...
```

```python
# inside _analyze_claude, after self._claude.messages.create(...)
langfuse_context.update_current_observation(
    model=self._claude_model,
    usage_details={"input": ..., "output": ...,
                   "cache_read_input_tokens": ...,
                   "cache_creation_input_tokens": ...},
    output=response.content[0].text,
)
```

- **Hosted on Langfuse Cloud US** (`https://us.cloud.langfuse.com`), as its own
  project — not the self-hosted Langfuse stack used by the OCI agent. The
  detector runs locally and shouldn't depend on the EC2 instance for trace
  delivery; Cloud's free tier covers personal-scale volume.
- **Opt-in via env vars.** When `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
  `LANGFUSE_HOST` are blank, the import-time fallback replaces `@observe` with
  a passthrough and `langfuse_context` with a no-op shim. No behavior change.
- **`capture_input=False`** on the decorator — the bound args include raw JPEG
  bytes; we hydrate `metadata` manually with sanitized scalars instead.
- `_raw_response` (the classifier path) also calls `update_current_observation`
  so `classify_person` traces show tokens — otherwise the highest-volume call
  in the pipeline would be the only one missing usage.

### Alternatives considered

- **Plain `logger.info` JSON per Claude call.** Skimmable. Doesn't diff
  prompt versions, doesn't render images, and isn't queryable across runs.
- **`langfuse.anthropic` drop-in client wrapper.** Less mature than the OpenAI
  shim in the v2 SDK; the documented decorator pattern is portable across
  SDK minor versions and avoids monkey-patching `anthropic.Anthropic()`.
- **Self-host this project's Langfuse on the same EC2 as OCI's.** Coupling
  the detector's observability to the EC2 instance was rejected — detector
  runs purely on the dev box, and the EC2 has unrelated downtime risk.

### Companion: pin `chromadb`

The rebuild that landed this change pulled `chromadb` 1.x for the first time
(spec was `>=0.5.0`); the new rust backend has a tighter SQL-parameter
ceiling that broke `EventVectorStore.__init__`'s single `get(limit=100_000)`
once the collection grew. Fixed two ways:

1. `vector_store.py` now paginates the hash-dedup load (500/page).
2. `requirements.txt` pins `chromadb>=1.5.9,<2.0.0` so the next rebuild
   doesn't surprise us with another major.

Documented here because it's the same rebuild and the lesson is the same:
**unpinned dependencies hide compatibility breaks until rebuild day, which
is usually also deploy day.**

## Consequences

**Positive**
- Per-call traces with system prompt, user prompt (or its hash for very long
  prompts), images, raw response, token usage, cache hits, and latency —
  inspectable long after `pipeline_trace` rotates.
- Prompt diffs in the Langfuse UI let us A/B prompt revisions on real images
  without re-running the dataset eval to compare.
- Cost slicing by metadata (`kind`, `backend`, `frames`) beats the
  Console-level "this whole key spent $X" view. Per-key Console attribution
  + per-trace Langfuse attribution stack cleanly.
- Tracing is fully optional — leaving env vars blank produces an identical
  binary path to the pre-ADR analyzer.

**Negative / watch out**
- Two Langfuse UIs to remember (OCI on self-hosted EC2, detector on Cloud US).
  Acceptable while only two projects use Langfuse; revisit if a third joins
  and the split starts costing context-switch.
- Trace traffic is buffered and flushed asynchronously. Long-running container
  is fine; short-lived scripts that call `analyze()` and exit should call
  `langfuse_context.flush()` before exit or lose the last batch.
- Adds `langfuse>=2.0,<4.0` to requirements. Cost of the rebuild that surfaced
  the chromadb 1.x compatibility break.
- `LANGFUSE_HOST` is the env var the SDK reads — not `LANGFUSE_BASE_URL`.
  Misnaming silently routes to the EU default cluster; verify with
  `docker exec ... sh -c 'echo $LANGFUSE_HOST'` after rotation.

## Principle

**A model call that drives a decision must be inspectable after the fact.**
[ADR-028](028-durable-decision-logging.md) made the outcome durable so the
system can't decide silently; this ADR makes the inputs durable so the
decision can't be re-derived only by re-running. Outcome logging and call
tracing are complementary: one tells you *what*, the other tells you *why*.
