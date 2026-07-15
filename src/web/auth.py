"""Phase 0 trust boundary — API-key auth for the FastAPI surface.

See docs/adr/030-aws-migration-strangler-fig.md (Phase 0) and
docs/adr/031-phase-0-api-key-auth.md for rationale.

Design decisions:
  * Single shared API key (env: PED_API_KEY). Multi-user / rotation is
    Phase 0.5. Fail-fast at startup if the env var is missing — we do
    NOT want the pipeline to run with an unauthenticated surface.
  * REST + static: bearer token in the Authorization header (or ?token=
    query param for browser <img> tags that can't send headers), checked
    with hmac.compare_digest to avoid timing side channels. Enforced by
    a single ASGI middleware — no per-route wiring, no route registry
    to keep in sync.
  * WebSocket: token via query param (browsers can't set headers on the
    WS handshake). Same compare_digest check via authorize_websocket().
  * Origin allowlist (env: PED_ALLOWED_ORIGINS, comma-separated) closes
    cross-site WebSocket hijacking (CSWSH) — browsers do NOT apply CORS
    to WS handshakes, so we check the Origin header ourselves.
  * The dashboard SPA (`/`, `/favicon.svg`, `/assets/*`) stays public so
    the login prompt can render. Snapshots + dataset images are behind
    auth. Everything else requires the key.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Callable, Iterable
from urllib.parse import parse_qs

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AuthConfig:
    """Resolved once at import time; blows up loudly if PED_API_KEY is unset."""

    def __init__(self) -> None:
        self.api_key = (os.getenv("PED_API_KEY") or "").strip()
        raw_origins = os.getenv("PED_ALLOWED_ORIGINS", "").strip()
        self.allowed_origins: frozenset[str] = frozenset(
            o.strip() for o in raw_origins.split(",") if o.strip()
        )
        if not self.api_key:
            raise RuntimeError(
                "PED_API_KEY is not set. Refusing to start with an "
                "unauthenticated surface. Set PED_API_KEY in .env "
                "(any long random string; e.g. `openssl rand -hex 32`)."
            )
        if len(self.api_key) < 16:
            raise RuntimeError(
                "PED_API_KEY is set but shorter than 16 chars. Use a "
                "high-entropy value: `openssl rand -hex 32`."
            )
        logger.info(
            "Auth enabled — key length=%d, allowed origins=%d (%s)",
            len(self.api_key),
            len(self.allowed_origins),
            ",".join(sorted(self.allowed_origins)) or "<none — same-origin only>",
        )


_CONFIG = AuthConfig()

# Paths served without auth so the SPA can load and prompt for a key.
_PUBLIC_PREFIXES: tuple[str, ...] = ("/assets/",)
_PUBLIC_EXACT: frozenset[str] = frozenset({"/", "/favicon.svg", "/health"})


def _check_token(candidate: str | None) -> bool:
    if not candidate:
        return False
    return hmac.compare_digest(candidate, _CONFIG.api_key)


def _origin_allowed(origin: str | None, host: str | None = None) -> bool:
    """Same-origin is allowed; cross-origin requires explicit allowlist.

    Same-origin means either (a) no Origin header at all (non-browser callers
    like curl) or (b) Origin matches the request's own Host header — browsers
    send Origin on WS handshakes and non-simple HTTP requests even when the
    target is the same page they loaded from.
    """
    if origin is None:
        return True
    if origin in _CONFIG.allowed_origins:
        return True
    if host:
        for scheme in ("http", "https"):
            if origin == f"{scheme}://{host}":
                return True
    return False


def _is_public_path(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


def _extract_token(headers: dict[str, str], query_string: bytes) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    # Fallback for <img src="/snapshots/..."> — browsers can't set headers there.
    qs = parse_qs(query_string.decode("latin-1"))
    vals = qs.get("token")
    return vals[0] if vals else None


# ── ASGI middleware ──────────────────────────────────────────────────────────

class AuthMiddleware:
    """Guards every HTTP request except _PUBLIC_PREFIXES / _PUBLIC_EXACT.

    Registered via `app.add_middleware(AuthMiddleware)`. WebSocket
    handshakes are NOT gated here — WS is a separate ASGI scope type and
    the response mechanism differs; use authorize_websocket() inside the
    handler.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if _is_public_path(path):
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }

        if not _origin_allowed(headers.get("origin"), headers.get("host")):
            await _send_status(send, 403, "origin not allowed")
            return

        token = _extract_token(headers, scope.get("query_string", b""))
        if not _check_token(token):
            await _send_status(send, 401, "unauthorized")
            return

        await self.app(scope, receive, send)


async def _send_status(send, code: int, message: str) -> None:
    body = message.encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": code,
        "headers": [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode("latin-1")),
            (b"www-authenticate", b"Bearer"),
        ],
    })
    await send({"type": "http.response.body", "body": body})


# ── WebSocket helper ──────────────────────────────────────────────────────────

async def authorize_websocket(websocket: WebSocket) -> bool:
    """Call BEFORE websocket.accept(). Closes the socket with 1008 on failure.

    Returns True when the handshake is authorized, False when it was rejected
    (caller should return without further work).

    Auth surface:
      * Origin header must be in the allowlist (or absent for same-origin).
      * ?token=<PED_API_KEY> must match.
    """
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not _origin_allowed(origin, host):
        logger.warning(
            "WS handshake rejected — origin %r not allowed (host=%r, allowlist=%d entries)",
            origin, host, len(_CONFIG.allowed_origins),
        )
        await websocket.close(code=1008, reason="origin not allowed")
        return False

    token = websocket.query_params.get("token")
    if not _check_token(token):
        logger.warning("WS handshake rejected — missing/invalid token")
        await websocket.close(code=1008, reason="unauthorized")
        return False

    return True


# ── confirm=true guard for destructive routes ────────────────────────────────

def require_confirm(confirm: bool = False) -> None:
    """FastAPI dependency — 400 unless ?confirm=true is present.

    Defense in depth beyond auth: even an authenticated caller shouldn't
    be able to nuke the dataset via a stray fetch(). The frontend admin
    UI must set confirm=true explicitly.
    """
    from fastapi import HTTPException, status
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Destructive operation requires ?confirm=true",
        )


__all__ = [
    "AuthMiddleware",
    "authorize_websocket",
    "require_confirm",
]
