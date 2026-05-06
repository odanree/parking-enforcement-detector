"""FastAPI web application.

Endpoints:
  GET  /              → dashboard HTML
  GET  /api/stats     → pipeline stats JSON
  GET  /api/events    → recent alert events JSON
  GET  /api/zone      → current street_zone polygon
  POST /api/zone      → update zone (persists to config/detection.yaml)
  WS   /ws/video      → MJPEG-over-WebSocket annotated frame stream (~15 fps)

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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
from src.web.state import AppState

state = AppState()
_BASE = Path(__file__).parent
_DETECTION_CFG = Path("config/detection.yaml")
_PRIVACY_CFG   = Path("config/privacy.json")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _seed_zone() -> None:
    cfg = _load_yaml(_DETECTION_CFG)
    polygon = cfg.get("zones", {}).get("street_zone", {}).get("polygon", [])
    state.update_zone(polygon)


def _seed_privacy() -> None:
    if _PRIVACY_CFG.exists():
        try:
            data = json.loads(_PRIVACY_CFG.read_text(encoding="utf-8"))
            state.update_privacy_regions(data.get("regions", []))
        except Exception:
            logger.warning("Could not load privacy regions from %s", _PRIVACY_CFG)


def _save_privacy(regions: list[list[int]]) -> None:
    _PRIVACY_CFG.parent.mkdir(exist_ok=True)
    _PRIVACY_CFG.write_text(json.dumps({"regions": regions}), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_zone()
    _seed_privacy()
    t = threading.Thread(target=pipeline.run, args=(state,), daemon=True, name="pipeline")
    t.start()
    logger.info("Pipeline thread started")
    yield
    logger.info("Shutting down")


_SNAPSHOTS_DIR = Path("snapshots")
_SNAPSHOTS_DIR.mkdir(exist_ok=True)
_DIST = _BASE.parent.parent / "frontend-dist"

app = FastAPI(title="Parking Enforcement Detector", lifespan=lifespan)
app.mount("/assets",    StaticFiles(directory=str(_DIST / "assets")),  name="assets")
app.mount("/snapshots", StaticFiles(directory=str(_SNAPSHOTS_DIR)),    name="snapshots")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return FileResponse(str(_DIST / "index.html"))


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    return JSONResponse(state.get_stats())


@app.get("/api/events")
async def get_events():
    return JSONResponse(state.get_events())


@app.get("/api/zone")
async def get_zone():
    return JSONResponse({"polygon": state.zone_polygon})


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
    enabled = state.toggle_privacy()
    return {"privacy_mode": enabled}


@app.get("/api/pending")
async def get_pending():
    return JSONResponse({"jobs": state.get_pending_vlm()})


@app.get("/api/debug/rejected")
async def get_rejected():
    return JSONResponse({"items": state.get_rejected_vlm()})


@app.delete("/api/debug/rejected")
async def clear_rejected():
    state.clear_rejected_vlm()
    return {"ok": True}


@app.get("/api/privacy/regions")
async def get_privacy_regions():
    return {"regions": state.get_privacy_regions()}


class PrivacyRegionsPayload(BaseModel):
    regions: list[list[int]]


@app.post("/api/privacy/regions")
async def update_privacy_regions(body: PrivacyRegionsPayload):
    state.update_privacy_regions(body.regions)
    _save_privacy(body.regions)
    return {"regions": state.get_privacy_regions()}


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


@app.post("/api/zone")
async def update_zone(body: ZonePayload):
    if len(body.polygon) < 3:
        raise HTTPException(status_code=400, detail="Zone needs at least 3 points")

    # Hot-reload: pipeline picks this up within one frame
    state.update_zone(body.polygon)

    # Persist to YAML so it survives restart
    cfg = _load_yaml(_DETECTION_CFG)
    cfg["zones"]["street_zone"]["polygon"] = body.polygon
    with open(_DETECTION_CFG, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    logger.info("Zone updated: %s", body.polygon)
    return {"status": "ok", "polygon": body.polygon}


# ── WebSocket video stream ────────────────────────────────────────────────────

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    logger.debug("WS client connected")
    try:
        while True:
            frame = state.get_frame()
            if frame:
                await websocket.send_bytes(frame)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.debug("WS client disconnected")
    except Exception:
        logger.exception("WS error")
