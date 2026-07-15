# ADR 032 — Phase 0.5: HttpOnly session cookies + one-shot WS tickets

**Status:** Accepted
**Date:** 2026-07-14

## Context

[ADR-031](031-phase-0-api-key-auth.md) established Phase 0's trust boundary using a shared API key transmitted in three ways:

- **REST fetches:** `Authorization: Bearer <key>` header
- **Static resources** (`<img src="/dataset/...">`): `?token=<key>` query param (browsers can't set headers on resource-loader requests)
- **WebSocket handshakes:** `?token=<key>` query param (same reason)

Key was stored in the browser's `localStorage` and injected into `fetch` / `WebSocket` calls by a global interceptor installed in `main.tsx`.

The Phase 0 ADR called out both mechanisms as **accepted trade-offs**. A re-audit after PR #6 merged (2026-07-14T21-32) reflagged them as HIGH-severity active findings — same underlying facts, different eye — and added a MEDIUM for the same root cause on the WebSocket path:

- `api-key-in-query-string` — token in `?token=` reaches server access logs, `Referer` headers, browser history, and shared HAR exports
- `api-key-in-localstorage-xss` — bearer secret in `localStorage` is exfiltrable by any XSS or malicious dependency in one line
- `websocket-token-in-handshake-url` — WS handshake URL logged by every reverse proxy the same way REST URLs are

All three share one architectural root: **the API key is a long-lived, high-privilege secret being routed through channels that don't protect long-lived, high-privilege secrets.** Fixing the mechanism (channels + secret lifecycle), not the value, is the correct move.

## Decision

Replace the shared-key transport with a two-part mechanism:

- **Browser flow → HttpOnly session cookie.** A new `POST /api/auth/login` accepts the API key, validates it against `PED_API_KEY`, mints an opaque server-side session id, and sets it as an `HttpOnly`, `SameSite=Strict`, `Secure` (optional) cookie. Every subsequent same-origin request (`fetch`, `<img src>`, `<link href>`) carries the cookie automatically. JavaScript cannot read it — closes the XSS exfil vector.
- **WebSocket → one-shot ticket.** A new `POST /api/auth/ws-ticket` (requires an authenticated caller) returns a `{ticket: "<32 hex>"}` valid for 30 seconds and single-use. Frontend calls this immediately before opening the WS and passes it as `?ticket=...`. Even if the handshake URL leaks to an access log, the ticket is already consumed or expired.
- **Bearer token in `Authorization` header remains supported** for non-browser callers (curl, API scripts, CI). Headers don't appear in URLs and aren't stored client-side, so bearer doesn't share the leak profile of `?token=`. Query-string `?token=` support is removed entirely.

### Files touched

- `src/web/auth.py` — session store, ticket store, cookie helpers, updated `AuthMiddleware` (cookie → bearer precedence), updated `authorize_websocket` (ticket → bearer precedence).
- `src/web/app.py` — four new endpoints: `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/status`, `POST /api/auth/ws-ticket`. Added to public-path list where appropriate (login/logout/status public; ws-ticket authenticated).
- `frontend/src/lib/auth.ts` — rewritten. `installAuthInterceptors()` now just forces `credentials: 'same-origin'` on all same-origin fetches. `ensureLoggedIn()` prompts on boot and calls `/api/auth/login`. `openAuthedWebSocket()` mints a ticket then opens the WS. `authedUrl()` removed — same-origin cookie handles images.
- `frontend/src/main.tsx` — defers React render until `ensureLoggedIn()` resolves.
- `frontend/src/hooks/useVideoStream.ts` + `frontend/src/components/EventModal.tsx` — awaited `openAuthedWebSocket()` in place of `new WebSocket()`.
- `frontend/src/components/DatasetAdmin.tsx` + `ComparisonDrawer.tsx` + `PipelineKanban.tsx` — dropped `authedUrl()` wrapping (cookie handles it).
- `tests/conftest.py` — Playwright autouse fixture now POSTs to `/api/auth/login` via `page.request` so the test context holds a valid cookie across navigations.

### Configuration

New env var: `PED_COOKIE_SECURE` (default `false`). Set `true` when serving over HTTPS so the cookie is only sent on secure connections. Default off for local `http://localhost:8000` dev; some browsers refuse `Secure` cookies on plaintext even for localhost.

### Session store

In-memory dict keyed by 32-byte random session id, 24-hour TTL, purged lazily on each mint. **Container restart logs everyone out** — acceptable for a single-instance home deploy where restarts are rare and re-login is a two-second prompt. Multi-instance deploys (or Phase 1's Fargate move) will need Redis or a signed-JWT alternative; explicitly out of scope for this ADR.

### Ticket store

Same in-memory shape, 30-second TTL, single-use enforcement. Purged on each mint. A leaked ticket is worthless within 30 seconds even without any active revocation.

## Alternatives considered

- **Signed JWTs instead of opaque session ids.** Stateless, survives container restart, no server-side store. Rejected for Phase 0.5 because JWT hygiene (rotation, algorithm confusion, revocation on compromise) adds design surface disproportionate to a single-instance home-box deploy. Signed-JWT can slot in when the multi-instance story arrives.
- **Signed URLs for static resources (HMAC + expiry) instead of cookie.** Cleaner for use cases where images need to be shareable outside the session (e.g., embedded in outbound email alerts). Rejected as the Phase 0.5 default because it multiplies moving parts (signing key rotation, expiry tuning, per-image URL generation) for questionable benefit — the current UI only renders images inside the dashboard where the cookie is already sent. Reserve for Phase 1 if outbound alert integration needs it.
- **Double-submit CSRF token.** `SameSite=Strict` + Origin allowlist already blocks the CSRF attack scenarios that apply here (LAN neighbor visiting attacker page). Adding a CSRF token is defense-in-depth but not load-bearing. Skipped for scope.
- **Keep `?token=` for backward compat and only add cookies.** Rejected — the whole point is closing the query-string leak vector. Half-migrations are worse: they leave the vulnerable path open while adding complexity.
- **Multi-user + audit log now.** Named as `shared-key-no-rotation-no-audit` (LOW) in the audit backlog. Genuine improvement, but the design touches a users table, per-role permissions, and an audit-write path that reaches into every destructive endpoint. Explicitly deferred to a Phase 0.6 ADR when that work is scoped.

## Consequences

**Positive**

- Closes both new HIGHs from the 2026-07-14T21-32 audit rerun (`api-key-in-query-string`, `api-key-in-localstorage-xss`) plus the associated MEDIUM (`websocket-token-in-handshake-url`) — three findings in one architectural move.
- API key exists in the browser for ~2 seconds during login submit, then never again — no JavaScript-accessible store to XSS-exfil.
- WebSocket handshake URLs no longer carry a long-lived secret. Access logs are still logs, but a leaked ticket is a leaked spent ticket.
- Query-string authentication is removed entirely. `?token=` becomes a 401 with a clear reason in the middleware; documented as removed in ADR-031's supersession note.
- `SameSite=Strict` cookie + Origin allowlist together give real CSRF defense on state-changing endpoints without a separate token layer.

**Negative / watch out**

- **Restart logs everyone out.** Acceptable for a single-instance home box; a footgun for anyone standing up a demo they leave running for days. Documented in the runbook; multi-instance session persistence lives in Phase 1.
- **The initial login UX is still `window.prompt`.** Fine for one operator, ugly for a demo group. Proper login modal is a straightforward React refactor when someone cares about the UX — deliberately not in this PR.
- **Two async round trips per WebSocket open** — one for the ticket, one for the WS handshake. Adds ~20 ms of connection latency on LAN. Not noticeable to users; noted here so future perf work knows where to look.
- **Bearer token support retained for API scripts** — the localStorage-XSS story doesn't apply to server-side callers, but if a script leaks its `.env`, the bearer key is game over. Same footprint as before; no regression, just no improvement on that path.
- **`api-key-in-query-string` may re-appear as `bearer-key-in-headers` in an over-strict future audit** if the reviewer treats any long-lived secret as a finding regardless of channel. Push back with this ADR: the *transport* is what changed, and bearer-in-header is a categorically weaker leak vector than bearer-in-URL.

## Migration notes

- Existing operators whose browsers hold the Phase 0 `localStorage['ped.apiKey']` will see the new login prompt on next load. Old value is orphaned but harmless — no functional path reads it. Clear it manually if the disk-space matters (it doesn't).
- Curl scripts using `-H "Authorization: Bearer $KEY"` continue to work unchanged.
- Curl scripts using `?token=$KEY` on URLs (Phase 0 fallback) will start returning 401. Move to the header. This is a **breaking change** for that specific pattern; called out in the PR description.

## Cross-references

- Supersedes [ADR-031](031-phase-0-api-key-auth.md)'s browser-side mechanism (bearer + `?token=` + localStorage). Backend `AuthMiddleware` still enforces the same trust boundary, just with different accepted credentials.
- Closes 3 findings from the 2026-07-14T21-32 audit rerun. Backlog will auto-transition them to `resolved` on the next `/portfolio-audit` run.
- Phase 1 (event bus to AWS SNS via HMAC-signed webhook, per [ADR-030](030-aws-migration-strangler-fig.md)) is still unblocked. Session store portability is now the last known Phase 1 blocker; likely resolved with a JWT swap when the multi-instance story lands.
