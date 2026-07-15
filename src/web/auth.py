"""Phase 0.5 trust boundary — cookie-based sessions + WS tickets.

Supersedes the Phase 0 mechanism (bearer token + `?token=` query param in URLs)
which had two documented HIGHs: API key exposed via server access logs / browser
history / Referer, and API key stored in localStorage where XSS could exfil it.

See docs/adr/030 (Phase 0), docs/adr/031 (Phase 0 mechanism), and
docs/adr/032 (this file's rationale).

Design decisions:
  * Login endpoint validates `PED_API_KEY` and mints an opaque session id
    stored in an HttpOnly, SameSite=Strict cookie. localStorage is no longer
    used for auth state — closes api-key-in-localstorage-xss.
  * WebSocket handshakes require a one-shot ticket obtained via an authenticated
    REST call — closes api-key-in-query-string and websocket-token-in-handshake-url.
    The ticket is opaque (32 hex chars) and single-use with a 30s TTL, so even
    if it leaks to an access log it's already used or expired.
  * Bearer token support is retained for non-browser callers (curl, API scripts).
    The Authorization header does not appear in URLs and is not stored client-
    side, so it does not have the same leak profile as `?token=`.
  * Session store is in-memory. Container restart logs everyone out — acceptable
    for a single-instance home deploy. Multi-instance would need Redis (out of
    scope for Phase 0.5).
  * Origin allowlist + same-origin auto-allow via Host-header comparison is
    unchanged from Phase 0. Combined with SameSite=Strict cookies, provides
    CSRF defense-in-depth without a separate CSRF token.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import time
from typing import Callable, Optional
from urllib.parse import parse_qs

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_SESSION_TTL_SEC = 24 * 60 * 60  # sessions expire after 24 hours
_TICKET_TTL_SEC = 30             # ws tickets are short-lived


class AuthConfig:
    """Resolved once at import time; blows up loudly if PED_API_KEY is unset."""

    def __init__(self) -> None:
        self.api_key = (os.getenv("PED_API_KEY") or "").strip()
        raw_origins = os.getenv("PED_ALLOWED_ORIGINS", "").strip()
        self.allowed_origins: frozenset[str] = frozenset(
            o.strip() for o in raw_origins.split(",") if o.strip()
        )
        # Cookies flagged Secure won't be sent over http:// (even localhost on
        # some browsers). Default off so local dev works; production TLS deploys
        # should set PED_COOKIE_SECURE=true.
        self.cookie_secure = os.getenv("PED_COOKIE_SECURE", "false").lower() == "true"
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
            "Auth enabled — key length=%d, allowed origins=%d (%s), cookie_secure=%s",
            len(self.api_key),
            len(self.allowed_origins),
            ",".join(sorted(self.allowed_origins)) or "<none — same-origin only>",
            self.cookie_secure,
        )


_CONFIG = AuthConfig()

# In-memory stores. Reset on container restart (documented trade-off — see ADR-032).
_sessions: dict[str, dict] = {}   # session_id -> {"created_at": ts}
_tickets: dict[str, dict] = {}    # ticket -> {"created_at": ts, "used": bool}

# Paths served without auth so the SPA can load the login flow.
_PUBLIC_PREFIXES: tuple[str, ...] = ("/assets/",)
_PUBLIC_EXACT: frozenset[str] = frozenset({
    "/", "/favicon.svg", "/health",
    # Auth endpoints are public because they gate everything else.
    "/api/auth/login", "/api/auth/logout", "/api/auth/status",
})

SESSION_COOKIE_NAME = "ped_session"


# ── Utilities ────────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()


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


def _bearer_from_header(headers: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def _parse_cookie_header(header: str) -> dict[str, str]:
    """Minimal cookie parser — we only need one specific cookie."""
    out: dict[str, str] = {}
    for part in header.split(";"):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            out[k.strip()] = v.strip()
    return out


# ── Session lifecycle (public API for the login/logout endpoints) ────────────

def _purge_expired_sessions() -> None:
    cutoff = _now() - _SESSION_TTL_SEC
    stale = [sid for sid, entry in _sessions.items() if entry["created_at"] < cutoff]
    for sid in stale:
        del _sessions[sid]


def check_api_key(candidate: str | None) -> bool:
    """Exposed for the login endpoint."""
    return _check_token(candidate)


def create_session() -> str:
    _purge_expired_sessions()
    sid = secrets.token_hex(32)
    _sessions[sid] = {"created_at": _now()}
    return sid


def is_valid_session(session_id: str | None) -> bool:
    if not session_id:
        return False
    entry = _sessions.get(session_id)
    if not entry:
        return False
    if _now() - entry["created_at"] > _SESSION_TTL_SEC:
        _sessions.pop(session_id, None)
        return False
    return True


def destroy_session(session_id: str | None) -> None:
    if session_id:
        _sessions.pop(session_id, None)


def cookie_settings() -> dict:
    """Cookie kwargs for `response.set_cookie` on login."""
    return {
        "key": SESSION_COOKIE_NAME,
        "httponly": True,
        "secure": _CONFIG.cookie_secure,
        "samesite": "strict",
        "max_age": _SESSION_TTL_SEC,
        "path": "/",
    }


# ── WS ticket lifecycle (public API for the ticket endpoint) ─────────────────

def _purge_expired_tickets() -> None:
    cutoff = _now() - _TICKET_TTL_SEC
    stale = [t for t, entry in _tickets.items()
             if entry["created_at"] < cutoff or entry["used"]]
    for t in stale:
        _tickets.pop(t, None)


def create_ws_ticket() -> str:
    _purge_expired_tickets()
    ticket = secrets.token_hex(24)
    _tickets[ticket] = {"created_at": _now(), "used": False}
    return ticket


def _consume_ws_ticket(ticket: str | None) -> bool:
    if not ticket:
        return False
    entry = _tickets.get(ticket)
    if not entry or entry["used"]:
        return False
    if _now() - entry["created_at"] > _TICKET_TTL_SEC:
        _tickets.pop(ticket, None)
        return False
    entry["used"] = True
    return True


# ── ASGI middleware ──────────────────────────────────────────────────────────

class AuthMiddleware:
    """Guards every HTTP request except _PUBLIC_PREFIXES / _PUBLIC_EXACT.

    Authentication precedence:
      1. Session cookie (browser flow — set by POST /api/auth/login).
      2. Bearer token in Authorization header (API/curl callers).

    Query-string `?token=` is NOT supported — that vector leaked the key via
    access logs, browser history, and Referer headers (see ADR-032).
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

        cookies = _parse_cookie_header(headers.get("cookie", ""))
        if is_valid_session(cookies.get(SESSION_COOKIE_NAME)):
            await self.app(scope, receive, send)
            return

        if _check_token(_bearer_from_header(headers)):
            await self.app(scope, receive, send)
            return

        await _send_status(send, 401, "unauthorized")


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

    Returns True when the handshake is authorized, False when rejected (caller
    should return without further work).

    Auth surface:
      * Origin header must be in the allowlist (or absent for same-origin).
      * ?ticket=<opaque-one-shot> minted via POST /api/auth/ws-ticket. Ticket
        is single-use with a 30s TTL — a leaked handshake URL is already spent
        or expired by the time an attacker sees it.
      * Bearer token in Authorization header remains supported (for non-browser
        callers that can set headers on the WS upgrade).
    """
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not _origin_allowed(origin, host):
        logger.warning(
            "WS handshake rejected — origin %r not allowed (host=%r)",
            origin, host,
        )
        await websocket.close(code=1008, reason="origin not allowed")
        return False

    ticket = websocket.query_params.get("ticket")
    if _consume_ws_ticket(ticket):
        return True

    # Header-based bearer for non-browser callers.
    if _check_token(_bearer_from_header({
        k.lower(): v for k, v in websocket.headers.items()
    })):
        return True

    logger.warning("WS handshake rejected — missing/invalid ticket and no bearer")
    await websocket.close(code=1008, reason="unauthorized")
    return False


# ── confirm=true guard for destructive routes ────────────────────────────────

def require_confirm(confirm: bool = False) -> None:
    """FastAPI dependency — 400 unless ?confirm=true is present.

    Defense in depth beyond auth: even an authenticated caller shouldn't be
    able to nuke the dataset via a stray fetch(). The frontend admin UI must
    set confirm=true explicitly.
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
    "check_api_key",
    "cookie_settings",
    "create_session",
    "create_ws_ticket",
    "destroy_session",
    "is_valid_session",
    "require_confirm",
    "SESSION_COOKIE_NAME",
]
