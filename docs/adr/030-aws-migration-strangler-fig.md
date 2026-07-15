# ADR 030 — AWS migration via strangler-fig at the detection-event boundary

**Status:** Proposed
**Date:** 2026-07-14

## Context

Today the detector is a single Python process on one GPU host — RTSP ingest, YOLO, gates, VLM, ChromaDB RAG, dedup, alerts, and dashboard all in-process ([ADR-001](001-two-stage-detection-yolo-then-vlm.md), [ADR-007](007-vlm-async-threadpool.md), [ADR-021](021-chromadb-evaluation-vector-store.md), [ADR-028](028-durable-decision-logging.md)). A recent security + architecture audit against the codebase surfaced three pressures that this ADR responds to.

**1. The single-host trust boundary has failed.** The audit flagged 7 high-severity findings, all rooted in one cause: the FastAPI + WebSocket surface on `0.0.0.0:8000` has zero authentication. Anyone reachable at the box's IP can view live feeds, wipe the dataset (`DELETE /api/dataset` → `vector_store.clear_all()`), tamper with privacy zones, drive the physical PTZ, or fan out unlimited Claude calls via `POST /api/dataset/reeval` and burn the Anthropic key. The audit's own words: *"auth-in-front-of-app closes ~half the findings in one change."*

**2. The pipeline has no service boundaries and therefore no failure isolation.** The audit's architecture-patterns pass detected five patterns already in the code (provider-abstraction, bulkhead-on-VLM-pool, idempotency-key-on-phash, prompt-caching, pipes-and-filters) but none of the *cross-service resilience* patterns senior-engineering practice expects: no circuit-breaker on Bedrock/Anthropic 529s, no token-bucket rate limiter on the VLM path, no dead-letter queue for poison frames, no per-channel bulkhead on alert dispatch. When Claude throttles, the RTSP frame queue stalls to backpressure. When SES throttles, HA webhooks stall with it.

**3. The codebase has outgrown a single-host runtime.** Adding a second camera ([ADR-022](022-multi-camera-shared-state-architecture.md)) already pushed the shared-state design past what one Python process handles cleanly. Continuing to add capability — more cameras, richer alerts, larger vector store, historical replay — inside the monolith trades linearly with pain: every new feature increases the blast radius of a single restart, competes with inference for CPU, and shares fate with every other feature on the same event loop. Moving to distributed infra is the natural next step for a system that is now processing enough real-world state that the failure modes are architectural rather than local.

**Constraint that shapes the design.** GPU inference (YOLO + optional Ollama) is a continuous workload against sunk hardware. Cloud GPU for 24/7 RTSP costs ~$380/mo per camera (SageMaker g4dn.xlarge) vs. ~$0 marginal on the existing box. A full lift-and-shift is economically wrong; the migration must be a **hybrid edge/cloud split at the cost-and-latency boundary**, not a wholesale move.

## Decision

Strangler-fig migration in five phases. Each phase ships value independently and leaves the monolith runnable; the boundary line moves outward one stage at a time.

**Boundary-line rule.** The service boundary between edge and cloud is the JSON payload emitted when the pipeline commits to "person-near-vehicle detected worth escalating." Everything upstream of that event stays at the edge (GPU-bound, continuous). Everything downstream moves to elastic AWS infra.

### Phase 0 — Auth gate at the current boundary (prerequisite)

Before any event is published to cloud, the current monolith needs an authenticating reverse proxy in front. This closes the 7 high-severity findings from the audit and prevents the "publish detections outbound" path in Phase 1 from becoming a new unauthenticated egress vector.

- Caddy container in front of FastAPI, basic-auth or forward-auth to a lightweight session provider.
- WebSocket handshakes gated on Origin allowlist + auth token (closes `websocket-cswsh`).
- `/snapshots` and `/dataset` mounts move behind the same middleware (closes `surveillance-data-exposure`).
- Destructive endpoints (`DELETE /api/dataset`, `/api/pipeline/history`, etc.) require `confirm=true` in addition to auth (closes `destructive-endpoints-unauthenticated`).

**Not a full migration** — this is a security fix that also happens to establish the trust boundary the cloud phases depend on.

### Phase 1 — Extract the event bus (event-driven pub/sub)

Add a `CloudPublisher` alert channel to [src/alerts/notifier.py](../../src/alerts/notifier.py) that POSTs the detection event `{phash, snapshot_url, yolo_conf, ts, camera_id}` to an AWS API Gateway endpoint. The Lambda behind it HMAC-verifies (the edge signs with a shared secret), then publishes to SNS topic `ped.detections`.

- **AWS primitives:** API Gateway + Lambda + SNS.
- **Patterns demonstrated:** HMAC verification at the trust boundary, event-driven pub/sub, first Terraform module in this project's IaC tree.
- **Behavior change:** none — monolith still runs all downstream stages locally. Cloud is a passive observer that will incrementally take work.

### Phase 2 — Extract dedup (idempotency-key)

SQS subscriber to `ped.detections` → Lambda → DynamoDB `PutItem` with `ConditionExpression: attribute_not_exists(phash)` and 7-day TTL. The monolith calls a lightweight query API for the dedup verdict before firing local alerts; local phash tracking becomes fallback for edge-only mode.

- **AWS primitives:** SQS + Lambda + DynamoDB (with TTL).
- **Patterns demonstrated:** idempotency-key with conditional writes, first real service boundary crossed.
- **Migrates:** the persistent phash dedup at [src/storage/vector_store.py:104](../../src/storage/vector_store.py) and the in-memory people-alert dedup at [src/web/state.py:301](../../src/web/state.py).

### Phase 3 — Extract VLM confirm (rate limiter + circuit breaker + DLQ)

SQS consumer on ECS Fargate calls Bedrock Claude Sonnet 4.5 with the snapshot fetched from an S3 signed URL. Adds the three resilience patterns the monolith lacks:

- **Token-bucket rate limiter** via SQS `max-in-flight` and visibility timeout — Anthropic 529 no longer stalls the RTSP frame queue.
- **Circuit breaker** on Bedrock throttling — after N consecutive failures, service opens and routes to a `ped.detections.deferred` queue that drains when Bedrock is healthy again.
- **Dead-letter queue** for poison frames — after N retries, the message lands in `ped.detections.dlq` with a CloudWatch alarm; the frame feeds the dataset flywheel via [scripts/triage_yolo_false_positives.py](../../scripts/triage_yolo_false_positives.py) already in the repo.

Also closes the audit's `unauthenticated-claude-cost-dos` finding: the Bedrock call site is no longer reachable from the FastAPI surface at all, and a **CloudWatch billing alarm** at $50/mo scales the Fargate service to zero.

- **AWS primitives:** SQS + ECS Fargate + Bedrock + CloudWatch billing alarms + Secrets Manager.
- **Patterns demonstrated:** token-bucket rate limiter, circuit breaker, DLQ for poison messages, cost budget with kill switch.
- **Monolith becomes:** edge fallback when `VLM_BACKEND=ollama` (offline mode retained).

### Phase 4 — RAG + dashboard split (read/write path separation)

- ChromaDB → RDS Aurora Serverless v2 with pgvector, retrieval service on Fargate. Closes the `embedding-model-versioned` finding by storing model+version per row; on read-time mismatch, trigger a reindex job.
- Dashboard becomes static S3 + CloudFront reading from an API Gateway WebSocket API backed by Lambda + DynamoDB Streams.

- **AWS primitives:** RDS Aurora Serverless v2, S3, CloudFront, API Gateway WebSocket.
- **Patterns demonstrated:** managed vector DB, CQRS-lite (writes flow through the event pipeline; reads come from a materialized DynamoDB view), scale-to-zero via Aurora Serverless.

### Phase 5 — Alert fan-out (per-channel bulkhead)

Replace [src/alerts/notifier.py](../../src/alerts/notifier.py) synchronous multi-channel dispatch with SNS topic `ped.alerts` + per-channel SQS subscribers, each with its own Lambda (SES, ntfy, HA webhook). An SES throttle no longer blocks HA.

- **AWS primitives:** SNS + N × SQS + N × Lambda.
- **Patterns demonstrated:** per-channel bulkhead / blast-radius isolation, saga pattern with compensating notification on partial delivery failure.

### Alternatives considered

- **Full lift-and-shift to SageMaker + Fargate.** Rejected — cloud GPU 24/7 for one camera is ~$380/mo against a $0-marginal-cost home box. Economics only work with N > 5 cameras or bursty (non-streaming) inference. Documented for later reactivation when scale changes.
- **Rewrite as a monolith on Fargate with an ALB.** Rejected — moves the compute but retains all seven audit findings' shape (single trust boundary, no failure isolation, no per-service scaling). Trades one hosting substrate for another without addressing the architectural failure modes this ADR exists to solve.
- **Bedrock AgentCore + Neptune for the whole pipeline.** Rejected — AgentCore is AWS's newer agent-runtime story and Neptune is its graph DB, but neither fits a CV data-flow pipeline. The supervisor pattern doesn't apply; PED is one-shot per detection, not multi-turn reasoning over a knowledge graph.
- **Skip Phase 0 auth and treat the migration as the security fix.** Rejected — the audit's high findings are exploitable *today* and the Phase 1 outbound-publish path becomes a new egress vector for surveillance data if the surface stays unauthenticated. Auth first.

### Cost + demo strategy

- **Scale-to-zero everywhere:** Fargate services `min=0`, Aurora Serverless v2 `min=0.5 ACU`, Bedrock pay-per-token, DynamoDB on-demand. Idle target: **< $15/mo**.
- **Demo mode:** `scripts/replay_from_video.py` publishes canned events into the SNS ingest so reviewers can hit the dashboard URL without a live RTSP feed.
- **Kill switch:** CloudWatch billing alarm at $50/mo → SNS → Lambda that scales all Fargate services to zero. Same-shape as the pattern proposed in Phase 3.

### Terraform module layout

Standalone Terraform tree with its own remote state backend (S3 + DynamoDB lock). No shared state with other projects — keeps blast radius scoped to PED.

```
terraform/ped/
├── main.tf                  # backend, providers, tags
├── environments/
│   ├── prod/
│   └── staging/             # only up during demo runs
└── modules/
    ├── event-ingest/        # API GW + Lambda + SNS (Phase 1)
    ├── dedup/               # DynamoDB + Lambda (Phase 2)
    ├── vlm-confirm/         # ECS + SQS + Bedrock IAM + DLQ (Phase 3)
    ├── rag/                 # Aurora Serverless v2 + retrieval service (Phase 4)
    ├── alerts/              # SNS + N x SQS + N x Lambda (Phase 5)
    ├── dashboard/           # S3 + CloudFront + WS API (Phase 4)
    └── observability/       # CloudWatch dashboards, X-Ray, billing alarms
```

## Consequences

**Positive**

- Closes the seven high-severity audit findings by moving the vulnerable surface (destructive endpoints, Claude cost, live snapshots) off the unauthenticated FastAPI box entirely — the boundary line becomes the SNS topic, not port 8000.
- Adds four resilience patterns to the codebase as first-class code — token-bucket rate limiter, circuit breaker, DLQ for poison messages, per-channel bulkhead — each landing in Phase 3 or Phase 5 and each addressing a concrete failure mode the current monolith exhibits under load.
- GPU inference stays where GPU already lives — no cost regression, no latency regression on the hot path (YOLO still runs at the edge).
- Each phase is independently shippable and reversible; the strangler pattern means the monolith is never broken mid-flight.
- The demo mode + scale-to-zero design keeps the live AWS deployment at < $15/mo idle, so the stack can stay up between demos without ongoing cost pressure to tear it down.

**Negative / watch out**

- Introduces an SPOF at the edge → cloud egress: if the home box loses internet, cloud stops receiving events. Acceptable because the local pipeline continues to fire local alerts (email/ntfy still work), but the dashboard goes stale. Document in the runbook.
- Adds a second observability plane (CloudWatch alongside Langfuse per [ADR-029](029-langfuse-vlm-observability.md)). Langfuse continues to trace individual VLM calls at the call site; CloudWatch traces the service topology. They don't overlap, but two dashboards to check.
- Phase 3's move of the VLM to Bedrock loses the offline `VLM_BACKEND=ollama` path unless the edge fallback is deliberately preserved. Explicitly keep the local backend wired as fallback so the pipeline degrades gracefully during AWS outages — this is **fail-fast at the trust boundary + graceful degradation on the runtime path**: the cloud path errors loudly and fast when Bedrock is unreachable, the edge Ollama backend takes over so detection continues, and the mismatch is surfaced via a CloudWatch alarm rather than a stalled pipeline.
- Terraform sprawl: nine modules is a lot. Phased delivery keeps this manageable — modules land as their phase does, not all at once — but the review discipline must hold.

## Principle

**A monolith becomes a distributed system the moment it needs a real trust boundary.** The audit forced the question: if you have to put an auth boundary somewhere anyway, the cheapest place to draw it is at the natural service boundary the code already has — the detection event. Once that boundary exists, the resilience patterns (rate limiter, circuit breaker, DLQ, bulkhead) become code that runs, not diagrams in an ADR; the security fix and the architectural improvement are the same move.

## Cross-references

- Related ADRs: [ADR-002](002-configurable-vlm-backend.md) (provider abstraction — Phase 3 preserves this), [ADR-007](007-vlm-async-threadpool.md) (bulkhead — Phase 3 externalizes), [ADR-021](021-chromadb-evaluation-vector-store.md) (ChromaDB choice — Phase 4 migrates), [ADR-029](029-langfuse-vlm-observability.md) (Langfuse — Phase 3 must preserve trace continuity across the boundary).
- Companion audit findings (7 high, 5 medium, 2 low, 1 info) are tracked out-of-tree and not published with this ADR; the seven high-severity items each map to a fix that Phase 0 or Phase 3 delivers.
