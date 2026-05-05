"""Parking enforcement detector — main pipeline.

Pipeline per frame:
  1. Grab frame from RTSP (TCP transport, drop stale frames)
  2. Run YOLOv8 tracker → person / truck / motorcycle at ≥65 % confidence
  3. Apply zone filter → keep detections inside the street_zone polygon
  4. Chalking check → if a person's bounding-box height drops ≥30 %, crop
     and send to VLM; fire alert when VLM confirms chalking.
  5. Sweeper check → only inside the configured sweep window; if a vehicle
     holds 5–10 mph velocity for 10+ frames, send full frame to VLM; fire
     alert when VLM confirms sweeper features.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.alerts.notifier import Notifier
from src.behavior.chalking_analyzer import ChalkingAnalyzer
from src.behavior.sweeper_analyzer import SweeperAnalyzer
from src.detection.object_detector import ObjectDetector
from src.detection.zone_filter import ZoneFilter
from src.stream.rtsp_handler import RTSPHandler
from src.vlm.analyzer import VLMAnalyzer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/detector.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 20) -> bytes:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def main() -> None:
    cfg_det = _load_yaml("config/detection.yaml")
    cfg_sched = _load_yaml("config/schedule.yaml")
    cfg_alerts = _load_yaml("config/alerts.yaml")

    # ── Component setup ───────────────────────────────────────────────────────
    stream = RTSPHandler(url=os.environ["RTSP_URL"])

    det_cfg = cfg_det["detector"]
    mask_cfg = cfg_det.get("stationary_mask", {})
    detector = ObjectDetector(
        model_path=os.getenv("YOLO_MODEL", det_cfg["model"]),
        threshold=float(os.getenv("INFERENCE_THRESHOLD", det_cfg["threshold"])),
        input_size=(
            int(os.getenv("INPUT_WIDTH", det_cfg["input_width"])),
            int(os.getenv("INPUT_HEIGHT", det_cfg["input_height"])),
        ),
        min_area_fraction=det_cfg["min_area_fraction"],
        max_area_fraction=det_cfg["max_area_fraction"],
        stationary_px=mask_cfg.get("pixel_threshold", 15),
        stationary_frames=mask_cfg.get("frames", 30),
    )

    zone_filter = ZoneFilter(
        zones={
            name: data["polygon"]
            for name, data in cfg_det["zones"].items()
        }
    )

    chalk_cfg = cfg_det["chalking"]
    chalking = ChalkingAnalyzer(
        height_decrease_threshold=chalk_cfg["height_decrease_threshold"],
        history_frames=chalk_cfg["history_frames"],
        cooldown_seconds=chalk_cfg["cooldown_seconds"],
    )

    sweep_cfg = cfg_det["sweeper"]
    sweeper = SweeperAnalyzer(
        schedule=cfg_sched["sweeper_schedule"],
        min_velocity=sweep_cfg["min_velocity_px_per_frame"],
        max_velocity=sweep_cfg["max_velocity_px_per_frame"],
        sustained_frames=sweep_cfg["sustained_frames"],
    )

    vlm = VLMAnalyzer(
        backend=os.getenv("VLM_BACKEND", "claude"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q4_K_M"),
    )

    notifier = Notifier(
        config=cfg_alerts["alerts"],
        ha_webhook_base=os.getenv("HA_WEBHOOK_URL", "").rsplit("/api/", 1)[0],
        ha_token=os.getenv("HA_TOKEN", ""),
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    stream.start()
    logger.info("Pipeline running — press Ctrl-C to stop")
    in_sweep_window = False

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                continue

            detections = detector.detect(frame)
            zone_dets = zone_filter.filter(detections, "street_zone")

            # Refresh sweep-window flag once per loop to avoid datetime overhead
            in_sweep_window = sweeper.is_sweep_window()

            active_ids = {d.track_id for d in zone_dets}

            for det in zone_dets:
                # ── Chalking (person only) ────────────────────────────────────
                if det.class_name == "person":
                    if chalking.update(det.track_id, det.height):
                        crop_bytes = _crop(frame, det.bbox)
                        result = vlm.analyze(crop_bytes)
                        if result["chalking_detected"]:
                            notifier.send("chalking", result, frame, det.bbox)

                # ── Sweeper (truck / vehicle, schedule-gated) ─────────────────
                elif det.class_name in {"truck", "motorcycle"} and in_sweep_window:
                    if sweeper.update(det.track_id, det.center):
                        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result = vlm.analyze(buf.tobytes() if ok else b"")
                        if result["sweeper_detected"]:
                            notifier.send("sweeper", result, frame, det.bbox)

            # Evict tracks that left the frame to keep memory bounded
            for tid in list(chalking._heights.keys()):
                if tid not in active_ids:
                    chalking.evict(tid)
            for tid in list(sweeper._centers.keys()):
                if tid not in active_ids:
                    sweeper.evict(tid)

    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
