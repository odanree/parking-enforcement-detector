# ADR 031 — Phase 0: API-key auth as the trust boundary

**Status:** Accepted
**Date:** 2026-07-14

## Context

[ADR-030](030-aws-migration-strangler-fig.md) established that a security + architecture audit surfaced 7 high-severity findings on the FastAPI + WebSocket surface — all rooted in one cause: no authentication anywhere. That ADR named Phase 0 as a prerequisite ("Caddy container in front, basic-auth or forward-auth to a lightweight session provider") but deliberately left the specific mechanism open. This ADR is that decision.

The audit's 7 highs and the shape of a fix for each:

- `missing-authentication` — every REST + WS endpoint open to `0.0.0.0:8000`
- `websocket-cswsh` — WS handshake accepts any Origin
- `unauthenticated-ptz-control` — physical camera movement callable by anyone
- `prompt-injection-and-cost-abuse` — `POST /api/vlm/prompt` accepts arbitrary prompts on the owner's Anthropic key
- `unauthenticated-claude-cost-dos` — `POST /api/dataset/reeval` fans out unbounded Claude calls
- `surveillance-data-exposure` — `/snapshots` and `/dataset` mounts served world-readable
- `destructive-endpoints-unauthenticated` — `DELETE /api/dataset` etc. wipe state with no confirmation

All seven share the same root: no trust boundary exists between the LAN and the app. Any mechanism that establishes one closes all seven at once.

## Decision

**FastAPI-native API-key auth via a single ASGI middleware, plus a per-handler WebSocket check.** No new container in the docker-compose. Implementation in [src/web/auth.py](../../src/web/auth.py); wired at [src/web/app.py](../../src/web/app.py).

### Mechanism

1. **Shared secret in env: `PED_API_KEY`.** The app refuses to start if it's unset or shorter than 16 chars — fail-fast at the trust boundary is preferable to a silent unauthenticated deploy.
2. **REST + static: bearer token in the `Authorization` header** for `fetch()` callers, or `?token=` query param for browser resource loaders (`<img>`, `<a href>` downloads) that can't set headers. Constant-time comparison via `hmac.compare_digest` avoids timing side channels.
3. **WebSocket: `?token=` query param** on the handshake URL, checked by `authorize_websocket()` before `websocket.accept()`. Handshakes fail closed with code 1008.
4. **Origin allowlist: `PED_ALLOWED_ORIGINS` env var**, comma-separated. Same-origin (no `Origin` header) is always allowed; cross-origin requests must be in the allowlist. This closes `websocket-cswsh` (browsers don't apply CORS to WS handshakes so we check `Origin` ourselves) and provides defense-in-depth against CSRF on state-changing REST routes.
5. **`confirm=true` guard on destructive DELETEs** (`/api/dataset`, `/api/pipeline/history`) — a FastAPI `Depends()` that 400s unless the caller explicitly opts in. Defense in depth beyond auth: even an authenticated caller shouldn't nuke the vector store via a stray `fetch()`.
6. **Frontend interceptors installed in [frontend/src/main.tsx](../../frontend/src/main.tsx) before any React component mounts.** Monkey-patches `window.fetch` and `window.WebSocket` at boot so the 18 existing `fetch()` call sites work unchanged. A helper `authedUrl()` handles the resource-loader case for image tags.

### Public paths

Kept unauthenticated so the login prompt can render and health probes work:

- `GET /` — the SPA entry HTML
- `GET /favicon.svg`
- `/assets/*` — the compiled React bundle
- `GET /health` — Docker liveness probe

Everything else — including `/snapshots/*` and `/dataset/*` static mounts — requires the key.

### Alternatives considered

- **Caddy reverse proxy with basic-auth in front.** Rejected for Phase 0. Adds a new container, requires DNS/TLS story for anything beyond `localhost`, and *still* requires FastAPI-side WS auth (Caddy can proxy WS but can't inspect the handshake body cleanly for token validation). All the setup cost, none of the WS coverage. Reserved for Phase 0b when internet exposure becomes a real need — Caddy handles TLS termination naturally at that point.
- **OAuth-proxy (oauth2-proxy in front, upstream trust via headers).** Rejected — overkill for a single-operator home box. Requires an IdP (Google/GitHub/Authelia), adds container and DNS complexity, and the WS handshake trust boundary is still awkward.
- **Session cookies + login form + CSRF tokens.** Rejected as Phase 0 because it multiplies the surface (login endpoint, session store, CSRF token generation, rotation logic) for no additional security over a bearer key. The bearer-token model is already what a session cookie *is* under the hood; skipping the UX layer keeps the fix small. Sessions can slot in as Phase 0.5 when there's demand for multi-user or auto-logout.
- **Bearer via header ONLY (no query-param fallback).** Rejected because `<img src="/dataset/foo.jpg">` cannot carry an `Authorization` header — the browser's resource loader owns that request, not our JS. Skipping the query-param path would either force us to blob-fetch every image (heavy) or leave images broken (broken).
- **Trust the LAN, don't add auth at all.** Rejected — this is what the audit found. A shared network is not a trust boundary; smart-home devices, guest Wi-Fi, and VPN pivots all violate the assumption.

## Consequences

**Positive**

- All 7 high-severity audit findings close with this one change. The 2 mediums flagged by the audit (`privacy-config-tamper`, `csrf-state-changing-post`) also close as byproducts of the middleware + Origin allowlist.
- Fail-fast startup makes the "accidentally deployed without auth" failure mode impossible — the box either has a key or won't run.
- Zero-config for the frontend: 18 existing `fetch()` call sites unchanged; only the image URL constructors got a `authedUrl()` wrap.
- No new container in the docker-compose — the runtime is still `ollama + detector`. Debugging the auth path is one Python module, one traceback.
- Sets up the Phase 1 outbound-egress work: the same PED_API_KEY becomes the identity that signs HMAC payloads when publishing detection events to AWS.

**Negative / watch out**

- **API key visible in browser URL bar** for `<img src="/dataset/foo.jpg?token=…">` — anyone with over-the-shoulder access to a demo can see it. Acceptable for a home box; rotate the key after any live demo. Not acceptable long-term — Phase 0.5 (session cookies or signed URLs) fixes this.
- **API key stored in `localStorage`** — vulnerable to XSS. There is no untrusted user-content path in this app today, so the XSS surface is minimal, but the risk is real. Rotate on any dependency-update audit that flags an XSS vector.
- **No rate limiting yet.** An attacker with the key can still DoS `/api/dataset/reeval`. `slowapi` middleware is Phase 0.5 — deliberately skipped here to keep the diff small and reviewable.
- **No key rotation UX.** Change the env var, restart the container, and every browser has to re-enter the key. Fine for one operator; painful for a demo group.
- **`prompt()` for the initial key entry is ugly.** Acceptable placeholder; a proper login modal is a straightforward React refactor when someone cares about the UX.

## Cross-references

- [ADR-030](030-aws-migration-strangler-fig.md) — the strangler-fig migration this is Phase 0 of. Phase 1 (event bus) can now proceed safely.
- Audit findings closed: `missing-authentication`, `websocket-cswsh`, `unauthenticated-ptz-control`, `prompt-injection-and-cost-abuse`, `unauthenticated-claude-cost-dos`, `surveillance-data-exposure`, `destructive-endpoints-unauthenticated`, `privacy-config-tamper`, `csrf-state-changing-post`.
- Audit findings NOT closed by this ADR (tracked for Phase 0.5): `no-rate-limiting`, `secret-in-outbound-log` (RTSP creds in URL), `unbounded-response-size` on `/api/timeline`, `email-credential-in-env`, `yaml-write-race`, `log-rotation-only-safeguard`.
