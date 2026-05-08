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
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import json
import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/detector.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from src import pipeline
from src.storage.vector_store import EventVectorStore
from src.vlm.analyzer import VLMAnalyzer, _LENIENT_USER_PROMPT, _LENIENT_SYSTEM_PROMPT
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
    # potential positives pass through to Claude for final confirmation.
    _is_two_stage = bool(_confirm_backend) and _confirm_backend != _vlm_backend
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
            ollama_model=_ollama_model,
        )
        logger.info("Two-stage VLM: %s (lenient) → %s (strict)", _vlm_backend, _confirm_backend)
    else:
        _confirm_vlm = None

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
        logger.info("Pipeline-1 thread started (rtsp=%s video=%s)", cam1_rtsp, cam1_video)
    else:
        logger.info("No RTSP_URL_2 / VIDEO_PATH_2 set — camera 1 inactive")

    yield
    logger.info("Shutting down")


_SNAPSHOTS_DIR = Path("snapshots")
_SNAPSHOTS_DIR.mkdir(exist_ok=True)
_DIST = _BASE.parent.parent / "frontend-dist"

app = FastAPI(title="Parking Enforcement Detector", lifespan=lifespan)
app.mount("/assets",    StaticFiles(directory=str(_DIST / "assets")),  name="assets")
app.mount("/snapshots", StaticFiles(directory=str(_SNAPSHOTS_DIR)),    name="snapshots")
app.mount("/dataset",   StaticFiles(directory=str(_DATASET_DIR)),      name="dataset")


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


@app.delete("/api/debug/rejected")
async def clear_rejected():
    for s in states:
        s.clear_rejected_vlm()
    return {"ok": True}


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
async def dataset_list(offset: int = 0, limit: int = 50):
    return JSONResponse(vector_store.get_all(offset=offset, limit=limit))


class LabelPayload(BaseModel):
    label: str   # "true_positive" | "false_positive" | "true_negative" | "false_negative" | ""


@app.post("/api/dataset/{event_id}/label")
async def dataset_label(event_id: str, body: LabelPayload):
    try:
        vector_store.update_label(event_id, body.label)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Event not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/similar/{event_id}")
async def dataset_similar(event_id: str, n: int = 10):
    return JSONResponse({"items": vector_store.query_similar(event_id, n=n)})


@app.get("/api/dataset/export")
async def dataset_export():
    return JSONResponse(vector_store.export_labeled())


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
async def reeval_dataset(background_tasks: BackgroundTasks, ids: list[str] | None = None):
    """Re-evaluate stored events using REEVAL_BACKEND (default: claude).

    POST with no body → re-evaluate all events that have frame files on disk.
    POST with JSON body ["id1", "id2", ...] → re-evaluate specific events only.

    Runs in the background.  Poll GET /api/dataset/comparison for results.
    """
    if _reeval_progress["running"]:
        raise HTTPException(status_code=409, detail="Re-evaluation already in progress")

    vlm = _get_reeval_vlm()
    backend = os.getenv("REEVAL_BACKEND", "claude")
    all_items = vector_store.get_all(limit=10_000)["items"]
    targets = [e for e in all_items if not ids or e["id"] in ids]

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
async def playback_preview_ws(websocket: WebSocket, timestamp: float, camera_id: int = 0):
    """Stream NVR playback frames directly to the caller without touching the pipeline."""
    from src.stream.rtsp_handler import build_nvr_playback_url
    await websocket.accept()
    base_url = os.getenv("RTSP_URL_2" if camera_id == 1 else "RTSP_URL", "")
    ch_env   = os.getenv("AMCREST_CHANNEL_2" if camera_id == 1 else "AMCREST_CHANNEL")
    nvr_ch   = int(ch_env) if ch_env else None
    playback_url = build_nvr_playback_url(timestamp, base_url, nvr_channel=nvr_ch)
    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(None, _open_preview_cap, playback_url)
    try:
        while True:
            frame_bytes = await loop.run_in_executor(None, _read_preview_frame, cap)
            if frame_bytes is None:
                break
            await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(1 / 25)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Playback preview WS error")
    finally:
        await loop.run_in_executor(None, cap.release)


@app.websocket("/ws/video/{cam_id}")
async def video_stream(websocket: WebSocket, cam_id: int = 0):
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
