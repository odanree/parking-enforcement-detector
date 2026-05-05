"""FastAPI web application.

Endpoints:
  GET  /              → dashboard HTML
  GET  /api/stats     → pipeline stats JSON
  GET  /api/events    → recent alert events JSON
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

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    return templates.TemplateResponse("index.html", {"request": request})


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    return JSONResponse(state.get_stats())


@app.get("/api/events")
async def get_events():
    return JSONResponse(state.get_events())


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
            # ~15 fps; the pipeline pushes at the same rate so we never stall
            await asyncio.sleep(0.067)
    except WebSocketDisconnect:
        logger.debug("WS client disconnected")
    except Exception:
        logger.exception("WS error")
