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

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _seed_zone() -> None:
    cfg = _load_yaml(_DETECTION_CFG)
    polygon = cfg.get("zones", {}).get("street_zone", {}).get("polygon", [])
    state.update_zone(polygon)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_zone()
    t = threading.Thread(target=pipeline.run, args=(state,), daemon=True, name="pipeline")
    t.start()
    logger.info("Pipeline thread started")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Parking Enforcement Detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


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
            await asyncio.sleep(0.067)
    except WebSocketDisconnect:
        logger.debug("WS client disconnected")
    except Exception:
        logger.exception("WS error")
