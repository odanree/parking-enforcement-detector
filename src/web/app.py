"""FastAPI web application.

Endpoints:
  GET  /              → dashboard HTML
  GET  /api/stats     → pipeline stats JSON
  GET  /api/events    → recent alert events JSON
  GET  /api/zone      → current street_zone polygon
  POST /api/zone      → update zone (persists to config/detection.yaml)
  WS   /ws/video/{cam_id}  → MJPEG-over-WebSocket annotated frame stream (~15 fps)

Run with:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import base64 as _b64
import json
import yaml
import numpy as _np
import cv2 as _cv2
from datetime import datetime as _dt
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()


def _phash(thumbnail_b64: str | None) -> str | None:
    """8×8 average perceptual hash → 16-char hex string. Returns None on failure."""
    if not thumbnail_b64:
        return None
    try:
        data = _b64.b64decode(thumbnail_b64)
        arr  = _np.frombuffer(data, _np.uint8)
        img  = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = _cv2.resize(img, (8, 8), interpolation=_cv2.INTER_AREA).flatten()
        bits  = small > small.mean()
        return f'{int("".join("1" if b else "0" for b in bits), 2):016x}'
    except Exception:
        return None


def _parse_model_evals(raw: str | None) -> list[dict]:
    """Deserialise the model_evals JSON string stored in ChromaDB metadata."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


# ── PE enforcement window (mirrors pipeline.py — used for kanban routing) ─────
_PE_WINDOW_ENABLED = os.getenv("PE_WINDOW_ENABLED", "false").lower() == "true"
_PE_WINDOW_START   = int(os.getenv("PE_WINDOW_START", "8"))
_PE_WINDOW_END     = int(os.getenv("PE_WINDOW_END",   "16"))
# Playback mode: VIDEO_PATH is set → footage timestamps are historical, skip off-hours gate
_IS_PLAYBACK       = bool(os.getenv("VIDEO_PATH"))


def _is_out_of_hours(ts: float) -> bool:
    """Return True if the event timestamp falls outside the PE enforcement window.

    Always False in playback mode — footage timestamps are historical and shouldn't
    be filtered by the live enforcement window.
    """
    if _IS_PLAYBACK:
        return False
    h = _dt.fromtimestamp(ts).hour
    return not (_PE_WINDOW_START <= h < _PE_WINDOW_END)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Rotate so the log (incl. per-event DECISION lines) can't grow unbounded.
        RotatingFileHandler(
            "logs/detector.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        ),
    ],
)
logger = logging.getLogger(__name__)

from src import pipeline
from src.storage.vector_store import EventVectorStore
from src.vlm.analyzer import VLMAnalyzer, _LENIENT_USER_PROMPT, _LENIENT_SYSTEM_PROMPT
from src.web.auth import (
    AuthMiddleware,
    SESSION_COOKIE_NAME,
    authorize_websocket,
    check_api_key,
    cookie_settings,
    create_session,
    create_ws_ticket,
    destroy_session,
    is_valid_session,
    require_confirm,
)
from src.web.state import AppState

states = [AppState(0), AppState(1)]
state = states[0]   # alias for all cam-0 REST endpoints

# VLM instances — created once in lifespan, referenced by prompt hot-reload endpoints.
_primary_vlm: VLMAnalyzer | None = None
_confirm_vlm: VLMAnalyzer | None = None
_BASE = Path(__file__).parent
_DETECTION_CFG = Path("config/detection.yaml")
_PRIVACY_CFG   = Path("config/privacy.json")
_DATASET_DIR   = Path("dataset")
_DATASET_DIR.mkdir(exist_ok=True)

vector_store = EventVectorStore(db_path="data/vectors", dataset_path=str(_DATASET_DIR))


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_ZONE_KEYS = ["street_zone", "street_zone_1"]

def _seed_zone() -> None:
    cfg = _load_yaml(_DETECTION_CFG)
    for cam_id, zone_key in enumerate(_ZONE_KEYS):
        polygon = cfg.get("zones", {}).get(zone_key, {}).get("polygon", [])
        states[cam_id].update_zone(polygon)


def _privacy_cfg(cam_id: int) -> Path:
    # cam 0 falls back to legacy privacy.json so existing configs are preserved
    specific = Path(f"config/privacy_{cam_id}.json")
    if cam_id == 0 and not specific.exists() and _PRIVACY_CFG.exists():
        return _PRIVACY_CFG
    return specific


def _seed_privacy() -> None:
    for cam_id in range(len(states)):
        cfg_path = _privacy_cfg(cam_id)
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                states[cam_id].update_privacy_regions(data.get("regions", []))
            except Exception:
                logger.warning("Could not load privacy regions from %s", cfg_path)


def _save_privacy(cam_id: int, regions: list[list[int]]) -> None:
    path = Path(f"config/privacy_{cam_id}.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"regions": regions}), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _primary_vlm, _confirm_vlm
    _seed_zone()
    _seed_privacy()

    _vlm_backend = os.getenv("VLM_BACKEND", "claude")
    _confirm_backend = os.getenv("CONFIRM_BACKEND", "")
    _claude_model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    _ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    _ollama_model = os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q4_K_M")

    # Phase-1 (primary) uses the lenient prompt when Ollama is the backend so more
    # potential positives pass through to the confirm model.
    # Two-stage is active when confirm backend differs OR when both use ollama but
    # with different models (e.g. gemma4 → qwen2.5vl).
    _confirm_ollama_model = os.getenv("CONFIRM_OLLAMA_MODEL", _ollama_model)
    _same_backend_diff_model = (
        _confirm_backend == _vlm_backend == "ollama"
        and _confirm_ollama_model != _ollama_model
    )
    _is_two_stage = bool(_confirm_backend) and (_confirm_backend != _vlm_backend or _same_backend_diff_model)
    _primary_vlm = VLMAnalyzer(
        backend=_vlm_backend,
        claude_model=_claude_model,
        ollama_url=_ollama_url,
        ollama_model=_ollama_model,
        user_prompt=_LENIENT_USER_PROMPT if _is_two_stage and _vlm_backend == "ollama" else None,
        system_prompt=_LENIENT_SYSTEM_PROMPT if _is_two_stage and _vlm_backend == "ollama" else None,
    )
    if _is_two_stage:
        _confirm_vlm = VLMAnalyzer(
            backend=_confirm_backend,
            claude_model=_claude_model,
            ollama_url=_ollama_url,
            ollama_model=_confirm_ollama_model,
        )
        logger.info(
            "Two-stage VLM: %s/%s (lenient people-detect) → %s/%s (strict chalking)",
            _vlm_backend, _ollama_model, _confirm_backend, _confirm_ollama_model,
        )
    else:
        _confirm_vlm = None
        logger.info("Single-stage VLM: %s/%s (strict chalking)", _vlm_backend, _ollama_model)

    t0 = threading.Thread(
        target=pipeline.run,
        args=(states[0],),
        kwargs={"vector_store": vector_store, "vlm": _primary_vlm, "confirm_vlm": _confirm_vlm},
        daemon=True,
        name="pipeline-0",
    )
    t0.start()
    logger.info("Pipeline-0 thread started")

    cam1_rtsp  = os.getenv("RTSP_URL_2")
    cam1_video = os.getenv("VIDEO_PATH_2")
    if cam1_rtsp or cam1_video:
        t1 = threading.Thread(
            target=pipeline.run,
            kwargs={
                "state":        states[1],
                "stream_url":   cam1_rtsp or None,
                "video_path":   cam1_video or None,
                "zone_key":     "street_zone_1",
                "vector_store": vector_store,
                "vlm":          _primary_vlm,
                "confirm_vlm":  _confirm_vlm,
            },
            daemon=True,
            name="pipeline-1",
        )
        t1.start()
        import re as _re
        safe_rtsp = _re.sub(r'://[^:]+:[^@]+@', '://****:****@', cam1_rtsp or "")
        logger.info("Pipeline-1 thread started (rtsp=%s video=%s)", safe_rtsp, cam1_video)
    else:
        logger.info("No RTSP_URL_2 / VIDEO_PATH_2 set — camera 1 inactive")

    yield
    logger.info("Shutting down")


_SNAPSHOTS_DIR = Path("snapshots")
_SNAPSHOTS_DIR.mkdir(exist_ok=True)
_DIST = _BASE.parent.parent / "frontend-dist"

app = FastAPI(title="Parking Enforcement Detector", lifespan=lifespan)
# Phase 0 trust boundary — see docs/adr/030 + docs/adr/031.
# Middleware guards every HTTP route except `/`, `/favicon.svg`, `/assets/*`.
# WebSocket handlers call authorize_websocket() directly (separate ASGI scope).
app.add_middleware(AuthMiddleware)
app.mount("/assets",    StaticFiles(directory=str(_DIST / "assets")),  name="assets")
app.mount("/snapshots", StaticFiles(directory=str(_SNAPSHOTS_DIR)),    name="snapshots")
app.mount("/dataset",   StaticFiles(directory=str(_DATASET_DIR)),      name="dataset")


@app.get("/health")
async def health():
    """Unauthenticated liveness probe — public so docker healthchecks work."""
    return {"ok": True}


# ── Auth endpoints (Phase 0.5 — see docs/adr/032) ────────────────────────────

class LoginBody(BaseModel):
    api_key: str


@app.post("/api/auth/login")
async def auth_login(body: LoginBody, response: Response):
    """Validate PED_API_KEY and set an HttpOnly session cookie."""
    if not check_api_key(body.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    session_id = create_session()
    response.set_cookie(value=session_id, **cookie_settings())
    return {"ok": True}


@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    """Clear the session cookie and remove the server-side session."""
    destroy_session(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Public — used by the SPA on boot to decide whether to prompt for login."""
    session_ok = is_valid_session(request.cookies.get(SESSION_COOKIE_NAME))
    # Also honour a bearer token — lets a curl script call /api/auth/status too.
    auth = request.headers.get("authorization", "")
    bearer_ok = auth.lower().startswith("bearer ") and check_api_key(
        auth.split(" ", 1)[1].strip()
    )
    return {"authenticated": bool(session_ok or bearer_ok)}


@app.post("/api/auth/ws-ticket")
async def auth_ws_ticket():
    """Mint a short-lived, single-use ticket for a WebSocket handshake.

    Caller must be authenticated (session cookie or bearer). Ticket is
    consumed by authorize_websocket() on the WS upgrade; 30s TTL.
    """
    return {"ticket": create_ws_ticket()}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return FileResponse(str(_DIST / "index.html"))

@app.get("/favicon.svg")
async def favicon():
    return FileResponse(str(_DIST / "favicon.svg"), media_type="image/svg+xml")


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    s = state.get_stats()
    for cam in states[1:]:
        other = cam.get_stats()
        s["total_chalking"] += other["total_chalking"]
    return JSONResponse(s)


@app.get("/api/events")
async def get_events():
    events = []
    for cam_id, s in enumerate(states):
        for ev in s.get_events():
            events.append({**ev, "camera": cam_id})
    events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return JSONResponse(events)


class VotePayload(BaseModel):
    vote: str | None   # "up" | "down" | "archive" | null


@app.post("/api/events/{cam_id}/vote")
async def vote_event(cam_id: int, body: VotePayload, timestamp: float):
    if cam_id not in range(len(states)):
        raise HTTPException(status_code=404, detail="Unknown camera")
    try:
        found = states[cam_id].vote_event(timestamp, body.vote)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not found:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True}


@app.get("/api/zone/{cam_id}")
async def get_zone(cam_id: int = 0):
    if cam_id not in (0, 1):
        raise HTTPException(status_code=404, detail="Unknown camera")
    return JSONResponse({"polygon": states[cam_id].zone_polygon})


class ZonePayload(BaseModel):
    polygon: list[list[int]]


@app.post("/api/pipeline/pause")
async def toggle_pause():
    paused = state.toggle_pause()
    return {"paused": paused}


@app.post("/api/motion/toggle")
async def toggle_motion():
    enabled = state.toggle_motion_detect()
    return {"motion_detect_enabled": enabled}


@app.post("/api/privacy/toggle")
async def toggle_privacy():
    # Toggle all cameras together — one on/off switch for the whole system
    enabled = states[0].toggle_privacy()
    for s in states[1:]:
        s.privacy_mode = enabled
    return {"privacy_mode": enabled}


@app.get("/api/sessions")
async def get_sessions():
    sessions = []
    for cam_id, s in enumerate(states):
        for sess in s.get_sessions():
            sessions.append({**sess, "camera_id": cam_id})
    sessions.sort(key=lambda x: x["started_at"], reverse=True)
    return JSONResponse(sessions)


@app.get("/api/vlm/prompt")
async def get_vlm_prompt():
    return JSONResponse({
        "primary": _primary_vlm.get_prompts() if _primary_vlm else None,
        "confirm": _confirm_vlm.get_prompts() if _confirm_vlm else None,
    })


class PromptPayload(BaseModel):
    stage: str          # "primary" | "confirm"
    user_prompt: str | None = None
    system_prompt: str | None = None


@app.post("/api/vlm/prompt")
async def set_vlm_prompt(body: PromptPayload):
    if body.stage == "primary":
        if _primary_vlm is None:
            raise HTTPException(status_code=404, detail="Primary VLM not initialised")
        _primary_vlm.set_prompts(body.user_prompt, body.system_prompt)
    elif body.stage == "confirm":
        if _confirm_vlm is None:
            raise HTTPException(status_code=404, detail="Confirm VLM not initialised (no two-stage pipeline)")
        _confirm_vlm.set_prompts(body.user_prompt, body.system_prompt)
    else:
        raise HTTPException(status_code=400, detail="stage must be 'primary' or 'confirm'")
    return {"ok": True, "stage": body.stage}


@app.get("/api/pending")
async def get_pending():
    jobs = []
    for cam_id, s in enumerate(states):
        for job in s.get_pending_vlm():
            jobs.append({**job, "camera": cam_id})
    return JSONResponse({"jobs": jobs})


@app.get("/api/debug/rejected")
async def get_rejected():
    items = []
    for cam_id, s in enumerate(states):
        for item in s.get_rejected_vlm():
            items.append({**item, "camera": cam_id})
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return JSONResponse({"items": items})


@app.delete("/api/events")
async def clear_events():
    for s in states:
        s.clear_events()
    return {"ok": True}


@app.delete("/api/debug/rejected")
async def clear_rejected():
    for s in states:
        s.clear_rejected_vlm()
    return {"ok": True}


@app.delete("/api/pipeline/history", dependencies=[Depends(require_confirm)])
async def clear_pipeline_history():
    """Clear in-memory pipeline state (live kanban cards, events deque, alerts).

    Does NOT touch the persistent vector store / dataset — use DELETE /api/dataset
    for that.  This only clears the live view so the kanban resets to empty.
    Requires ?confirm=true.
    """
    for s in states:
        s.clear_all_history()
    return {"ok": True}


@app.get("/api/pipeline/trace")
async def pipeline_trace(
    limit: int = 500,
    since: float | None = None,
    label: str | None = None,
    camera_id: int | None = None,
):
    """Return pipeline trace cards for the kanban view — historical + live.

    Filters (all optional, default = last 7 days + unlabeled):
      since     — unix timestamp; only events at or after this time
      label     — '' = unlabeled only, 'true_positive' etc., omit = any
      camera_id — 0 or 1; omit = both cameras
    """
    # ── 1. Vector-store baseline (filtered historical events) ─────────────────
    stored = vector_store.get_filtered(
        since_ts=since,
        limit=limit,
        label=label,
        camera_id=camera_id,
        capture_source="chalking",
    )
    stored.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

    vs_cards: dict[str, dict] = {}
    for ev in stored:
        _model_primary = ev.get("model_primary", "") or "primary"
        _model_confirm = ev.get("model_confirm", "")
        first_pass = {
            "detected":    bool(ev.get("detected")),
            "confidence":  float(ev.get("confidence", 0)),
            "description": ev.get("description", ""),
            "backend":     _model_primary,
        }
        confirm = None
        if ev.get("reeval_backend"):
            confirm = {
                "detected":    bool(ev.get("reeval_detected")),
                "confidence":  float(ev.get("reeval_confidence", 0)),
                "description": ev.get("reeval_description", ""),
                "backend":     ev["reeval_backend"],
            }
        elif _model_confirm:
            confirm = None  # confirm stage was original pipeline — already captured in first_pass progression
        thumb_file   = ev.get("thumb_file", "")
        hires_file   = ev.get("hires_file", "")
        thumb_phash  = ev.get("thumb_phash") or None
        ev_ts        = float(ev.get("timestamp", 0))
        yolo_conf_ev = ev.get("yolo_confidence")
        card: dict = {
            "id":               ev["id"],
            "source":           "historical",
            "outcome":          "detected" if ev.get("detected") else "rejected",
            "timestamp":        ev_ts,
            "camera_id":        int(ev.get("camera_id", 0)),
            "confidence":       float(ev.get("confidence", 0)),
            "description":      ev.get("description", ""),
            "thumb_url":        f"/dataset/{thumb_file}" if thumb_file else None,
            "hires_url":        f"/dataset/{hires_file}" if hires_file else None,
            "thumbnail":        None,
            "phash":            thumb_phash,
            "track_id":         ev.get("track_id"),
            "rag_neighbors":    [],
            "rag":              None,
            "yolo":             {"confidence": yolo_conf_ev} if yolo_conf_ev is not None and yolo_conf_ev >= 0 else None,
            "first_pass":       first_pass,
            "confirm":          confirm,
            "label":            ev.get("label", ""),
            "person_type":      ev.get("person_type", ""),
            "capture_source":   ev.get("capture_source", "chalking"),
            "rejection_reason": None,
            "model_primary":    _model_primary,
            "model_confirm":    _model_confirm,
            "model_evals":      _parse_model_evals(ev.get("model_evals")),
        }
        vs_cards[ev["id"]] = card

    # ── 2. Live in-memory events (post-restart, have full pipeline_trace) ────
    live_ids: set[str] = set()
    live_cards: list[dict] = []

    for cam_id, s in enumerate(states):
        if camera_id is not None and cam_id != camera_id:
            continue
        for ev in s.get_events(limit=50):
            trace = ev.get("pipeline_trace")
            if trace is None:
                continue
            ev_ts = ev["timestamp"]
            if since is not None and ev_ts < since:
                continue
            live_cards.append({
                "id":               ev.get("db_id") or f"live-ev-{ev_ts}-{cam_id}",
                "source":           "live",
                "outcome":          "detected",
                "timestamp":        ev_ts,
                "camera_id":        cam_id,
                "confidence":       ev["confidence"],
                "description":      ev["description"],
                "thumb_url":        ev.get("snapshot_url"),
                "thumbnail":        ev["frames"][0] if ev.get("frames") else None,
                "phash":            _phash(ev["frames"][0] if ev.get("frames") else None),
                "rag_neighbors":    ev.get("rag_neighbors", []),
                "track_id":         ev.get("track_id"),
                "yolo":             trace.get("yolo"),
                "first_pass":       trace.get("first_pass"),
                "rag":              trace.get("rag"),
                "confirm":          trace.get("confirm"),
                "label":            "",
                "rejection_reason": None,
            })
        for item in s.get_rejected_vlm():
            item_ts = item["timestamp"]
            if since is not None and item_ts < since:
                continue
            trace = item.get("pipeline_trace")
            # Use only the explicitly recorded rejection_reason — no retroactive timestamp guessing.
            rejection_reason = item.get("rejection_reason")
            thumbnail = item.get("thumbnail") or ""
            # Include if: has VLM trace, has a known rejection reason, or has a snapshot thumbnail
            if trace is None and rejection_reason not in ("out_of_hours", "suppressed") and not thumbnail:
                continue
            live_cards.append({
                "id":               item.get("db_id") or f"live-rej-{item_ts}-{cam_id}",
                "source":           "live",
                "outcome":          "rejected",
                "timestamp":        item_ts,
                "camera_id":        cam_id,
                "confidence":       item["confidence"],
                "description":      item["description"],
                "thumb_url":        None,
                "hires_url":        item.get("hires_url"),
                "thumbnail":        thumbnail or None,
                "phash":            _phash(thumbnail or None),
                "rag_neighbors":    item.get("rag_neighbors", []),
                "track_id":         item.get("track_id"),
                "yolo":             trace.get("yolo") if trace else {
                    "confidence": item["confidence"],
                    "track_id":   item.get("track_id"),
                    "bbox":       item.get("bbox"),
                },
                "first_pass":       trace.get("first_pass") if trace else None,
                "rag":              trace.get("rag") if trace else None,
                "confirm":          trace.get("confirm") if trace else None,
                "label":            "",
                "rejection_reason": rejection_reason,
            })

    # ── 3. People-alert cards (two-stage: first-pass positive, pre-confirm) ──────
    for cam_id, s in enumerate(states):
        if camera_id is not None and cam_id != camera_id:
            continue
        for item in s.get_people_alerts():
            item_ts = item["timestamp"]
            if since is not None and item_ts < since:
                continue
            trace = item.get("pipeline_trace")
            live_cards.append({
                "id":               f"live-pa-{item_ts}-{cam_id}",
                "source":           "live",
                "outcome":          "people_alert",
                "timestamp":        item_ts,
                "camera_id":        cam_id,
                "confidence":       item["confidence"],
                "description":      item["description"],
                "thumb_url":        None,
                "thumbnail":        item.get("thumbnail"),
                "phash":            _phash(item.get("thumbnail")),
                "rag_neighbors":    item.get("rag_neighbors", []),
                "track_id":         item.get("track_id"),
                "yolo":             trace.get("yolo") if trace else None,
                "first_pass":       trace.get("first_pass") if trace else None,
                "rag":              trace.get("rag") if trace else None,
                "confirm":          trace.get("confirm") if trace else None,
                "label":            "",
                "rejection_reason": None,
            })

    # Merge: live cards take priority (have base64 thumbnails); vs_cards fill the rest.
    # Deduplicate by (camera_id, rounded timestamp) so the live version wins when both exist,
    # but backfill vs-only fields (hires_url, model info, person_type) into the live card.
    vs_by_key: dict[tuple, dict] = {
        (vc["camera_id"], round(vc["timestamp"])): vc for vc in vs_cards.values()
    }
    _VS_BACKFILL = ("hires_url", "model_primary", "model_confirm", "person_type", "model_evals", "capture_source", "label")
    for lc in live_cards:
        vc = vs_by_key.get((lc["camera_id"], round(lc["timestamp"])))
        if vc:
            lc["id"] = vc["id"]  # promote to real DB id so the live-dot clears
            for field in _VS_BACKFILL:
                if lc.get(field) is None and vc.get(field) is not None:
                    lc[field] = vc[field]
    live_keys     = {(c["camera_id"], round(c["timestamp"])) for c in live_cards}
    live_real_ids = {c["id"] for c in live_cards if not c["id"].startswith("live-")}
    all_cards = list(live_cards)
    for vc in vs_cards.values():
        if (vc["camera_id"], round(vc["timestamp"])) not in live_keys and vc["id"] not in live_real_ids:
            all_cards.append(vc)

    all_cards.sort(key=lambda c: c["timestamp"], reverse=True)
    return JSONResponse({
        "cards":      all_cards[:limit],
        "total":      len(all_cards),
        "_debug": {
            "vs_count":   len(vs_cards),
            "live_count": len(live_cards),
            "limit":      limit,
            "since":      since,
            "label":      label,
            "camera_id":  camera_id,
        },
    })


@app.get("/api/pipeline/trace/debug")
async def pipeline_trace_debug():
    """Quick diagnostic — returns vector store count and first 3 item IDs."""
    try:
        count = vector_store.count()
        sample = vector_store.get_all(limit=3)["items"]
        return JSONResponse({
            "count": count,
            "sample_ids": [e["id"] for e in sample],
            "sample_timestamps": [e.get("timestamp") for e in sample],
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/privacy/regions/{cam_id}")
async def get_privacy_regions(cam_id: int = 0):
    if cam_id not in range(len(states)):
        raise HTTPException(status_code=404, detail="Unknown camera")
    return {"regions": states[cam_id].get_privacy_regions()}


class PrivacyRegionsPayload(BaseModel):
    regions: list[list[int]]


_DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"


@app.post("/api/privacy/regions/{cam_id}")
async def update_privacy_regions(body: PrivacyRegionsPayload, cam_id: int = 0):
    if cam_id not in range(len(states)):
        raise HTTPException(status_code=404, detail="Unknown camera")
    if _DEMO_MODE:
        raise HTTPException(status_code=403, detail="Privacy regions are locked in demo mode")
    states[cam_id].update_privacy_regions(body.regions)
    _save_privacy(cam_id, body.regions)
    return {"regions": states[cam_id].get_privacy_regions()}


class SpeedPayload(BaseModel):
    speed: float


class DirectionPayload(BaseModel):
    direction: int  # 1 = forward, -1 = reverse

class SeekPayload(BaseModel):
    seconds: float


@app.post("/api/playback/speed")
async def set_playback_speed(body: SpeedPayload):
    speed = state.set_playback_speed(body.speed)
    return {"speed": speed}


@app.post("/api/playback/direction")
async def set_playback_direction(body: DirectionPayload):
    direction = state.set_playback_direction(body.direction)
    return {"direction": direction}

@app.post("/api/playback/seek")
async def seek_playback(body: SeekPayload):
    state.seek_playback(body.seconds)
    return {"ok": True}


class TimestampPayload(BaseModel):
    timestamp: float   # Unix epoch seconds


@app.post("/api/playback/seek-timestamp")
async def seek_to_timestamp(body: TimestampPayload):
    result = state.seek_to_timestamp(body.timestamp)
    if result != 'ok':
        raise HTTPException(status_code=409, detail=result)
    return {"ok": True}


@app.post("/api/playback/live")
async def go_live():
    result = state.go_live()
    if result != 'ok':
        raise HTTPException(status_code=409, detail=result)
    return {"ok": True}


# ── Dataset / labeling ────────────────────────────────────────────────────────

@app.get("/api/dataset")
async def dataset_list(
    offset: int = 0,
    limit: int = 50,
    label: str | None = None,
    camera_id: int | None = None,
    capture_source: str | None = None,
    exclude_source: str | None = None,
    detected: int | None = None,
):
    result = vector_store.get_filtered(
        offset=offset,
        limit=limit,
        label=label,
        camera_id=camera_id,
        capture_source=capture_source,
        exclude_source=exclude_source,
        detected=detected,
    )
    if isinstance(result, list):
        return JSONResponse({"total": len(result), "offset": offset, "limit": limit, "items": result})
    return JSONResponse(result)


@app.delete("/api/dataset", dependencies=[Depends(require_confirm)])
async def dataset_clear_all():
    """Wipe the entire vector store. Requires ?confirm=true in addition to auth."""
    removed = vector_store.clear_all()
    return {"ok": True, "removed": removed}


@app.delete("/api/dataset/{event_id}")
async def dataset_delete(event_id: str):
    try:
        vector_store.delete(event_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Event not found")


# ── PTZ control ───────────────────────────────────────────────────────────────

class PTZMovePayload(BaseModel):
    direction: str        # up / down / left / right / up_left / … / zoom_in / zoom_out
    speed: int = 4        # 1–8


class PTZStopPayload(BaseModel):
    direction: str | None = None


@app.post("/api/camera/{camera_id}/ptz/move")
async def api_ptz_move(camera_id: int, body: PTZMovePayload):
    from src.stream.ptz import ptz_move
    ok = await asyncio.get_event_loop().run_in_executor(
        None, lambda: ptz_move(camera_id, body.direction, body.speed)
    )
    return {"ok": ok}


@app.post("/api/camera/{camera_id}/ptz/stop")
async def api_ptz_stop(camera_id: int, body: PTZStopPayload = PTZStopPayload()):
    from src.stream.ptz import ptz_stop
    ok = await asyncio.get_event_loop().run_in_executor(
        None, lambda: ptz_stop(camera_id, body.direction)
    )
    return {"ok": ok}


@app.post("/api/camera/{camera_id}/ptz/preset/{preset}")
async def api_ptz_preset(camera_id: int, preset: int):
    from src.stream.ptz import ptz_preset
    ok = await asyncio.get_event_loop().run_in_executor(
        None, lambda: ptz_preset(camera_id, preset)
    )
    return {"ok": ok}


class LabelPayload(BaseModel):
    label: str   # "true_positive" | "false_positive" | "true_negative" | "false_negative" | ""


@app.post("/api/dataset/{event_id}/label")
async def dataset_label(event_id: str, body: LabelPayload):
    if event_id.startswith("live-"):
        raise HTTPException(status_code=422, detail="Card not yet persisted — wait for database index")
    try:
        vector_store.update_label(event_id, body.label)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Event not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class PersonTypePayload(BaseModel):
    person_type: str  # pedestrian | resident | occupant | worker_landscape | worker_delivery | chalker | ""


@app.post("/api/dataset/{event_id}/person-type")
async def dataset_person_type(event_id: str, body: PersonTypePayload):
    try:
        vector_store.update_person_type(event_id, body.person_type)
        return {"ok": True, "person_type": body.person_type}
    except KeyError:
        raise HTTPException(status_code=404, detail="Event not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset/{event_id}/capture-hires")
async def dataset_capture_hires(event_id: str):
    """Re-grab a hi-res frame for a historical event from the NVR recording.

    Works only while the footage for that timestamp is still on the NVR disk
    (within retention). Updates the event's hires_file and re-tags it
    capture_source=manual_hires.
    """
    if event_id.startswith("live-"):
        raise HTTPException(status_code=422, detail="Card not yet persisted")
    from src.stream.recorded_frame import fetch_recorded_frame
    rec = vector_store._col.get(ids=[event_id], include=["metadatas"])
    if not rec["ids"]:
        raise HTTPException(status_code=404, detail="Event not found")
    meta = rec["metadatas"][0]
    ts = float(meta.get("timestamp", 0) or 0)
    cam = int(meta.get("camera_id", 0) or 0)
    if ts <= 0:
        raise HTTPException(status_code=400, detail="Event has no timestamp")
    jpeg = await asyncio.to_thread(fetch_recorded_frame, cam, ts)
    if not jpeg:
        raise HTTPException(
            status_code=404,
            detail="No recorded footage at that time (out of retention or NVR unreachable)",
        )
    try:
        fname = vector_store.attach_hires(event_id, jpeg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store hi-res: {e}")
    return {"ok": True, "hires_url": f"/dataset/{fname}", "capture_source": "manual_hires"}


@app.get("/api/timeline")
async def api_timeline():
    """Return lightweight [{t, cam, det, people}] for every stored record, sorted oldest-first.

    det=1    → confirmed chalking
    people=1 → person detected by first-pass VLM but not confirmed as chalking
               (model_confirm is set = confirm stage ran = first pass was positive)
    both=0   → rejected with no person detected
    """
    result = vector_store._col.get(include=["metadatas"], limit=100_000)
    rows = []
    for meta in (result.get("metadatas") or []):
        t   = meta.get("timestamp")
        cam = meta.get("camera_id")
        det = meta.get("detected")
        if t is None or cam is None or det is None:
            continue
        # person detected if: confirm stage ran (two-stage) OR zone_pedestrian capture
        people = 1 if (
            meta.get("model_confirm") or
            meta.get("capture_source") == "zone_pedestrian"
        ) else 0
        yc = meta.get("yolo_confidence", -1.0)
        rows.append({"t": float(t), "cam": int(cam), "det": int(det), "people": people, "yc": float(yc)})
    rows.sort(key=lambda r: r["t"])
    return JSONResponse(rows)


class _TimelineAnalyzeRequest(BaseModel):
    rows: list[dict]


@app.post("/api/timeline/analyze")
async def api_timeline_analyze(req: _TimelineAnalyzeRequest):
    """Stream a Claude security-expert analysis of the timeline data via SSE."""
    import anthropic

    rows = req.rows
    if not rows:
        return JSONResponse({"error": "No timeline data provided"}, status_code=400)

    # Build a concise statistical summary for the prompt
    total = len(rows)
    chalking = sum(1 for r in rows if r.get("det"))
    people   = sum(1 for r in rows if not r.get("det") and r.get("people"))
    high_yolo = sum(1 for r in rows if not r.get("det") and r.get("yc", -1) >= 0.70)
    rejected  = total - chalking - people - high_yolo

    # Time range
    ts_sorted = sorted(r["t"] for r in rows)
    from datetime import datetime as _datetime
    t_start = _datetime.fromtimestamp(ts_sorted[0]).strftime("%Y-%m-%d %H:%M:%S")
    t_end   = _datetime.fromtimestamp(ts_sorted[-1]).strftime("%Y-%m-%d %H:%M:%S")
    span_h  = (ts_sorted[-1] - ts_sorted[0]) / 3600

    # Gap detection (>30 min)
    GAP_MIN = 30 * 60
    gaps = []
    for i in range(1, len(ts_sorted)):
        dur = ts_sorted[i] - ts_sorted[i - 1]
        if dur > GAP_MIN:
            gaps.append({
                "start": _datetime.fromtimestamp(ts_sorted[i - 1]).strftime("%Y-%m-%d %H:%M"),
                "end":   _datetime.fromtimestamp(ts_sorted[i]).strftime("%Y-%m-%d %H:%M"),
                "dur_h": round(dur / 3600, 1),
            })

    # Camera breakdown
    cams: dict[int, dict] = {}
    for r in rows:
        c = int(r.get("cam", 0))
        bucket = cams.setdefault(c, {"total": 0, "chalking": 0})
        bucket["total"] += 1
        if r.get("det"):
            bucket["chalking"] += 1

    cam_lines = "\n".join(
        f"  Camera {c}: {v['total']} snapshots, {v['chalking']} chalking events"
        for c, v in sorted(cams.items())
    )
    gap_lines = "\n".join(
        f"  {g['start']} → {g['end']} ({g['dur_h']}h gap)"
        for g in gaps
    ) or "  None detected"

    summary = f"""Timeline data summary:
- Period: {t_start} to {t_end} ({span_h:.1f} hours)
- Total snapshots: {total}
- Confirmed chalking events: {chalking}
- People detected (VLM positive, no chalking): {people}
- High-confidence YOLO detections (≥70%, not chalking): {high_yolo}
- Rejected / no person: {rejected}
- Cameras:
{cam_lines}
- Coverage gaps (>30 min with no snapshots):
{gap_lines}"""

    system_prompt = (
        "You are a parking enforcement timeline security expert. "
        "You analyze snapshot detection data from a dual-camera parking enforcement system "
        "that uses YOLO object detection and a VLM (Vision Language Model) to identify chalk-marking events. "
        "Your job is to interpret detection patterns, flag anomalies, estimate system health, "
        "and surface any operational concerns — such as coverage gaps, unusual detection ratios, "
        "potential camera issues, or suspicious activity windows. "
        "Be direct, specific, and concise. Use bullet points. Avoid generic advice."
    )

    async def stream_response():
        client = anthropic.Anthropic()
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": summary}],
        ) as stream:
            for text in stream.text_stream:
                # SSE format: data: <chunk>\n\n
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/api/dataset/similar/{event_id}")
async def dataset_similar(event_id: str, n: int = 10):
    return JSONResponse({"items": vector_store.query_similar(event_id, n=n)})


@app.get("/api/dataset/export")
async def dataset_export():
    return JSONResponse(vector_store.export_labeled())


@app.post("/api/dataset/deduplicate")
async def dataset_deduplicate():
    removed = vector_store.deduplicate()
    return {"removed": removed, "remaining": vector_store.count()}


# ── Re-evaluation (second-opinion) ───────────────────────────────────────────

_reeval_vlm: VLMAnalyzer | None = None


def _get_reeval_vlm() -> VLMAnalyzer:
    global _reeval_vlm
    if _reeval_vlm is None:
        backend = os.getenv("REEVAL_BACKEND", "claude")
        _reeval_vlm = VLMAnalyzer(
            backend=backend,
            claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("REEVAL_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "")),
        )
        logger.info("Re-eval VLM initialised (backend=%s)", backend)
    return _reeval_vlm


_reeval_progress: dict = {"running": False, "done": 0, "total": 0, "errors": 0}


@app.post("/api/dataset/reeval")
async def reeval_dataset(background_tasks: BackgroundTasks, ids: list[str] | None = None, detected_only: bool = False):
    """Re-evaluate stored events using REEVAL_BACKEND (default: claude).

    POST with no body → re-evaluate all events that have frame files on disk.
    POST with JSON body ["id1", "id2", ...] → re-evaluate specific events only.
    ?detected_only=true → only re-evaluate events the original model flagged as detected.

    Runs in the background.  Poll GET /api/dataset/comparison for results.
    """
    if _reeval_progress["running"]:
        raise HTTPException(status_code=409, detail="Re-evaluation already in progress")

    vlm = _get_reeval_vlm()
    backend = os.getenv("REEVAL_BACKEND", "claude")
    all_items = vector_store.get_all(limit=10_000)["items"]
    targets = [e for e in all_items if not ids or e["id"] in ids]
    if detected_only:
        targets = [e for e in targets if e.get("detected")]

    def _run() -> None:
        _reeval_progress.update(running=True, done=0, total=len(targets), errors=0)
        for ev in targets:
            frames = vector_store.get_frame_bytes(ev["id"])
            if not frames:
                _reeval_progress["errors"] += 1
                continue
            try:
                result = vlm.analyze(frames, "chalking")
                vector_store.update_reeval(
                    ev["id"],
                    backend=backend,
                    detected=result["chalking_detected"],
                    confidence=result["confidence"],
                    description=result["description"],
                )
            except Exception:
                logger.exception("reeval failed for event %s", ev["id"])
                _reeval_progress["errors"] += 1
            _reeval_progress["done"] += 1
        _reeval_progress["running"] = False
        logger.info(
            "Re-eval complete: %d/%d done, %d errors",
            _reeval_progress["done"], _reeval_progress["total"], _reeval_progress["errors"],
        )

    background_tasks.add_task(_run)
    return {"queued": len(targets), "backend": backend}


class ModelEvalRequest(BaseModel):
    model_name: str                       # e.g. "qwen2.5vl:7b"
    backend: str = "ollama"               # "ollama" | "claude"
    task: str = "detect"                  # "detect" = chalking yes/no | "classify" = person_type
    ids: list[str] | None = None          # None = all events with frames on disk
    detected_only: bool = False           # only re-run events primary model flagged
    person_type_filter: str | None = None # only re-run events with this person_type label


_model_eval_progress: dict = {
    "running": False, "done": 0, "total": 0, "errors": 0, "model": "", "task": "",
}


@app.post("/api/dataset/model-eval")
async def run_model_eval(body: ModelEvalRequest, background_tasks: BackgroundTasks):
    """Retroactively score stored events with any named model.

    task="detect"   — asks model "is this chalking?" → stores detected/confidence/description
    task="classify" — asks model "what type of person?" → stores person_type/confidence/description

    Results are appended to each event's model_evals list (idempotent per model+task key).
    Poll GET /api/dataset/model-eval/progress for status.
    """
    if _model_eval_progress["running"]:
        raise HTTPException(status_code=409, detail="Model eval already in progress")
    if body.task not in ("detect", "classify"):
        raise HTTPException(status_code=400, detail="task must be 'detect' or 'classify'")

    ollama_url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    eval_vlm = VLMAnalyzer(
        backend=body.backend,
        claude_model=claude_model,
        ollama_url=ollama_url,
        ollama_model=body.model_name if body.backend == "ollama" else "",
        # Use lenient detection prompt for detect task; classify_person() overrides prompts itself
        user_prompt=_LENIENT_USER_PROMPT if body.task == "detect" and body.backend == "ollama" else None,
        system_prompt=_LENIENT_SYSTEM_PROMPT if body.task == "detect" and body.backend == "ollama" else None,
    )

    all_items = vector_store.get_all(limit=10_000)["items"]
    targets = [e for e in all_items if not body.ids or e["id"] in body.ids]
    if body.detected_only:
        targets = [e for e in targets if e.get("detected")]
    if body.person_type_filter is not None:
        targets = [e for e in targets if e.get("person_type", "") == body.person_type_filter]

    # Key used in model_evals to distinguish detect vs classify runs of the same model
    eval_key = f"{body.model_name}:{body.task}"

    def _run() -> None:
        _model_eval_progress.update(
            running=True, done=0, total=len(targets), errors=0,
            model=body.model_name, task=body.task,
        )
        for ev in targets:
            frames = vector_store.get_frame_bytes(ev["id"])
            if not frames:
                _model_eval_progress["done"] += 1
                continue
            try:
                if body.task == "classify":
                    result = eval_vlm.classify_person(frames)
                    vector_store.add_model_eval(
                        ev["id"],
                        model_name=eval_key,
                        backend=body.backend,
                        detected=int(result.get("person_type", "unknown") not in ("unknown", "")),
                        confidence=result["confidence"],
                        description=f'[{result["person_type"]}] {result["description"]}',
                    )
                else:
                    result = eval_vlm.analyze(frames, "chalking")
                    vector_store.add_model_eval(
                        ev["id"],
                        model_name=eval_key,
                        backend=body.backend,
                        detected=result["chalking_detected"],
                        confidence=result["confidence"],
                        description=result["description"],
                    )
            except Exception:
                logger.exception("model-eval failed for event %s", ev["id"])
                _model_eval_progress["errors"] += 1
            _model_eval_progress["done"] += 1
        _model_eval_progress["running"] = False
        logger.info(
            "Model eval %s (%s) complete: %d/%d done, %d errors",
            body.model_name, body.task,
            _model_eval_progress["done"], _model_eval_progress["total"], _model_eval_progress["errors"],
        )

    background_tasks.add_task(_run)
    return {"queued": len(targets), "model": body.model_name, "backend": body.backend, "task": body.task}


@app.get("/api/dataset/model-eval/progress")
async def model_eval_progress():
    return JSONResponse(_model_eval_progress)


@app.get("/api/dataset/comparison")
async def comparison_report():
    """Return all events that have a re-eval result alongside the original.

    Each item includes:
      detected / confidence / description   — original model result
      reeval_detected / reeval_confidence / reeval_description / reeval_backend — second opinion
      agreement — true if both models agree on detected flag
    """
    all_items = vector_store.get_all(limit=10_000)["items"]
    compared  = [e for e in all_items if "reeval_backend" in e]
    for ev in compared:
        ev["agreement"] = bool(ev["detected"]) == bool(ev["reeval_detected"])

    total      = len(compared)
    agreements = sum(1 for e in compared if e["agreement"])
    return JSONResponse({
        "progress":       _reeval_progress,
        "total":          total,
        "agreement_rate": round(agreements / total, 3) if total else None,
        "disagreements":  [e for e in compared if not e["agreement"]],
        "agreements":     [e for e in compared if e["agreement"]],
    })


@app.get("/api/pipeline/model-stats")
async def model_stats():
    """Per-model performance statistics vs YOLO detections.

    Returns a breakdown per model name showing:
      - total events processed
      - detection rate (how often the model confirmed a person/event)
      - average VLM confidence
      - label breakdown (TP/FP/TN/FN/unlabeled) when manually labeled
      - YOLO confidence buckets vs VLM detection rate
    """
    all_items = vector_store.get_all(limit=10_000)["items"]

    from collections import defaultdict

    def _model_bucket(items: list[dict]) -> dict:
        total = len(items)
        if not total:
            return {"total": 0}
        detected  = sum(1 for e in items if e.get("detected"))
        avg_conf  = sum(float(e.get("confidence", 0)) for e in items) / total

        label_counts: dict[str, int] = defaultdict(int)
        for e in items:
            label_counts[e.get("label") or "unlabeled"] += 1

        person_type_counts: dict[str, int] = defaultdict(int)
        for e in items:
            person_type_counts[e.get("person_type") or "unlabeled"] += 1

        # Detection rate broken down by human-assigned person_type
        by_person_type: dict[str, dict] = {}
        for pt in set(e.get("person_type") or "" for e in items):
            bucket = [e for e in items if (e.get("person_type") or "") == pt]
            if not bucket:
                continue
            b_det = sum(1 for e in bucket if e.get("detected"))
            by_person_type[pt or "unlabeled"] = {
                "total":          len(bucket),
                "detected":       b_det,
                "detection_rate": round(b_det / len(bucket), 3),
                "avg_confidence": round(sum(float(e.get("confidence", 0)) for e in bucket) / len(bucket), 3),
            }

        # YOLO confidence buckets vs VLM detection rate
        yolo_buckets: dict[str, dict] = {}
        for lo, hi, name in [
            (0.0, 0.30, "<0.30"),
            (0.30, 0.50, "0.30–0.50"),
            (0.50, 0.70, "0.50–0.70"),
            (0.70, 0.90, "0.70–0.90"),
            (0.90, 1.01, "≥0.90"),
        ]:
            bucket = [
                e for e in items
                if e.get("yolo_confidence") is not None
                and float(e["yolo_confidence"]) >= 0
                and lo <= float(e["yolo_confidence"]) < hi
            ]
            if bucket:
                b_det = sum(1 for e in bucket if e.get("detected"))
                yolo_buckets[name] = {
                    "total":          len(bucket),
                    "detected":       b_det,
                    "detection_rate": round(b_det / len(bucket), 3),
                }

        return {
            "total":           total,
            "detected":        detected,
            "detection_rate":  round(detected / total, 3),
            "avg_confidence":  round(avg_conf, 3),
            "labels":          dict(label_counts),
            "person_types":    dict(person_type_counts),
            "by_person_type":  by_person_type,
            "yolo_buckets":    yolo_buckets,
        }

    import json as _json

    by_model: dict[str, list[dict]] = defaultdict(list)
    for ev in all_items:
        model = ev.get("model_primary") or "unknown"
        by_model[model].append(ev)

    by_confirm: dict[str, list[dict]] = defaultdict(list)
    for ev in all_items:
        cm = ev.get("model_confirm") or ""
        if cm:
            by_confirm[cm].append(ev)

    # Retroactive model evals — build synthetic event dicts so _model_bucket() works
    by_eval_model: dict[str, list[dict]] = defaultdict(list)
    for ev in all_items:
        try:
            evals = _json.loads(ev.get("model_evals") or "[]")
        except Exception:
            evals = []
        for entry in evals:
            mn = entry.get("model") or "unknown"
            by_eval_model[mn].append({
                "detected":        entry.get("detected", 0),
                "confidence":      entry.get("confidence", 0.0),
                "label":           ev.get("label", ""),
                "person_type":     ev.get("person_type", ""),
                "yolo_confidence": ev.get("yolo_confidence"),
            })

    # Dataset composition summary
    person_type_summary: dict[str, int] = defaultdict(int)
    capture_source_summary: dict[str, int] = defaultdict(int)
    for ev in all_items:
        person_type_summary[ev.get("person_type") or "unlabeled"] += 1
        capture_source_summary[ev.get("capture_source") or "chalking"] += 1

    return JSONResponse({
        "total_events":         len(all_items),
        "dataset_composition":  {
            "by_person_type":   dict(person_type_summary),
            "by_capture_source": dict(capture_source_summary),
        },
        "primary_models":       {m: _model_bucket(items) for m, items in by_model.items()},
        "confirm_models":       {m: _model_bucket(items) for m, items in by_confirm.items()},
        "eval_models":          {m: _model_bucket(items) for m, items in by_eval_model.items()},
    })


class AlertPayload(BaseModel):
    event_type: str
    timestamp: float
    confidence: float
    description: str | None = None
    snapshot_url: str | None = None


@app.post("/api/alert")
async def send_alert(body: AlertPayload):
    from datetime import datetime, timezone
    import httpx

    to = os.getenv("ALERT_PHONE", "")
    # Strip + for TextBelt (expects digits only)
    to_digits = to.lstrip("+")

    ts_str     = datetime.fromtimestamp(body.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    type_label = {"chalking": "Chalking", "sweeper": "Sweeper", "pe_vehicle": "PE Vehicle"}.get(body.event_type, body.event_type)
    pct        = round((body.confidence or 0) * 100)
    desc       = (body.description or "No description")[:200]
    msg        = f"[PED Alert] {type_label} detected ({pct}%)\n{ts_str}\n{desc}"

    # Email — set ALERT_EMAIL, SMTP_USER, SMTP_PASS in .env
    alert_email = os.getenv("ALERT_EMAIL")
    smtp_user   = os.getenv("SMTP_USER")
    smtp_pass   = os.getenv("SMTP_PASS")
    if alert_email and smtp_user and smtp_pass:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage
        try:
            em = MIMEMultipart()
            em["Subject"] = f"PED Alert: {type_label} detected ({pct}%)"
            em["From"]    = smtp_user
            em["To"]      = alert_email
            em.attach(MIMEText(msg, "plain"))

            # Attach snapshot if available and on disk
            if body.snapshot_url:
                # snapshot_url is "/snapshots/<filename>"
                snap_path = _SNAPSHOTS_DIR / Path(body.snapshot_url).name
                if snap_path.exists():
                    em.attach(MIMEImage(snap_path.read_bytes(), name=snap_path.name))

            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.send_message(em)
            logger.info("Alert email sent to %s", alert_email)
            return {"ok": True}
        except Exception as exc:
            logger.error("Email alert failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # ntfy.sh push notification — set NTFY_TOPIC=your-topic in .env
    ntfy_topic = os.getenv("NTFY_TOPIC")
    if ntfy_topic:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"https://ntfy.sh/{ntfy_topic}",
                    content=msg.encode(),
                    headers={"Title": f"PED: {type_label} detected"},
                )
            logger.info("Alert sent via ntfy to topic %s", ntfy_topic)
            return {"ok": True}
        except Exception as exc:
            logger.error("ntfy alert failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # TextBelt SMS — set TEXTBELT_KEY to a paid key from textbelt.com
    textbelt_key = os.getenv("TEXTBELT_KEY")
    if textbelt_key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    "https://textbelt.com/text",
                    data={"phone": to_digits, "message": msg, "key": textbelt_key},
                )
            data = r.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "TextBelt failed"))
            logger.info("Alert SMS sent via TextBelt to %s", to)
            return {"ok": True}
        except Exception as exc:
            logger.error("TextBelt alert failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # Twilio fallback
    sid   = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_ = os.getenv("TWILIO_FROM_NUMBER")
    if not (sid and token and from_):
        raise HTTPException(
            status_code=503,
            detail="No SMS provider configured — set TEXTBELT_KEY or TWILIO_* vars in .env",
        )
    try:
        from twilio.rest import Client
        Client(sid, token).messages.create(body=msg, from_=from_, to=to)
        logger.info("Alert SMS sent via Twilio to %s", to)
        return {"ok": True}
    except Exception as exc:
        logger.error("Twilio alert failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/zone/{cam_id}")
async def update_zone(body: ZonePayload, cam_id: int = 0):
    if cam_id not in (0, 1):
        raise HTTPException(status_code=404, detail="Unknown camera")
    if len(body.polygon) < 3:
        raise HTTPException(status_code=400, detail="Zone needs at least 3 points")

    states[cam_id].update_zone(body.polygon)

    cfg = _load_yaml(_DETECTION_CFG)
    cfg["zones"][_ZONE_KEYS[cam_id]]["polygon"] = body.polygon
    with open(_DETECTION_CFG, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    logger.info("Zone %d updated: %s", cam_id, body.polygon)
    return {"status": "ok", "polygon": body.polygon}


# ── WebSocket video stream ────────────────────────────────────────────────────

def _open_preview_cap(url: str):
    import cv2
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _read_preview_frame(cap) -> bytes | None:
    import cv2
    ok, frame = cap.read()
    if not ok:
        return None
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


@app.websocket("/ws/playback/preview")
async def playback_preview_ws(websocket: WebSocket, timestamp: float, camera_id: int = 0, speed: int = 1):
    """Stream NVR playback frames directly to the caller without touching the pipeline.

    Speed is achieved client-side: a background thread drains the NVR stream as fast as
    the NVR allows; the WS loop skips (speed-1) frames per displayed frame at 25 fps.
    speedpara is still passed to the NVR URL in case the firmware honours it.
    """
    if not await authorize_websocket(websocket):
        return
    import collections
    from src.stream.rtsp_handler import build_nvr_playback_url
    await websocket.accept()
    base_url = os.getenv("RTSP_URL_2" if camera_id == 1 else "RTSP_URL", "")
    ch_env   = os.getenv("AMCREST_CHANNEL_2" if camera_id == 1 else "AMCREST_CHANNEL")
    nvr_ch   = int(ch_env) if ch_env else None
    speed    = max(1, min(8, speed))
    playback_url = build_nvr_playback_url(timestamp, base_url, nvr_channel=nvr_ch, speed=speed)
    loop = asyncio.get_event_loop()
    cap  = await loop.run_in_executor(None, _open_preview_cap, playback_url)

    # Background thread reads frames as fast as the NVR sends them into a bounded deque.
    # The display loop then skips (speed-1) frames per send, giving client-side speedup.
    buf      = collections.deque(maxlen=speed * 12)
    stop_evt = threading.Event()

    def _fill_buf():
        while not stop_evt.is_set():
            fb = _read_preview_frame(cap)
            if fb is None:
                break
            buf.append(fb)

    reader = threading.Thread(target=_fill_buf, daemon=True)
    reader.start()

    try:
        while True:
            # Wait for at least one frame
            if not buf:
                await asyncio.sleep(0.005)
                continue
            # Discard speed-1 frames to advance content faster
            for _ in range(speed - 1):
                if buf:
                    buf.popleft()
            if buf:
                await websocket.send_bytes(buf.popleft())
            await asyncio.sleep(1 / 25)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Playback preview WS error")
    finally:
        stop_evt.set()
        await loop.run_in_executor(None, cap.release)


@app.websocket("/ws/video/{cam_id}")
async def video_stream(websocket: WebSocket, cam_id: int = 0):
    if not await authorize_websocket(websocket):
        return
    if cam_id not in (0, 1):
        await websocket.close(code=1008)
        return
    cam_state = states[cam_id]
    await websocket.accept()
    logger.debug("WS client connected (cam %d)", cam_id)
    try:
        while True:
            frame = cam_state.get_frame()
            if frame:
                await websocket.send_bytes(frame)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.debug("WS client disconnected (cam %d)", cam_id)
    except Exception:
        logger.exception("WS error (cam %d)", cam_id)
