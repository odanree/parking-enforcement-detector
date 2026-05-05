"""Detection pipeline — runs in the main thread (headless) or a daemon thread
(when the web dashboard is active).

Call `run(state)` to start.  Pass an `AppState` instance so the web layer
can read annotated frames and events in real-time.  Pass `None` for headless.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.alerts.notifier import Notifier
from src.behavior.chalking_analyzer import ChalkingAnalyzer
from src.behavior.sweeper_analyzer import SweeperAnalyzer
from src.detection.object_detector import Detection, ObjectDetector
from src.detection.zone_filter import ZoneFilter
from src.stream.rtsp_handler import RTSPHandler
from src.vlm.analyzer import VLMAnalyzer

logger = logging.getLogger(__name__)

# Push at most this many frames per second to the WebSocket state.
_MAX_PUSH_FPS = 15
_PUSH_EVERY_N = max(1, 30 // _MAX_PUSH_FPS)   # assumes ~30 fps camera


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _annotate(
    frame: np.ndarray,
    all_dets: list[Detection],
    zone_dets: list[Detection],
    alert_ids: set[int],
) -> np.ndarray:
    out = frame.copy()

    zone_id_set = {d.track_id for d in zone_dets}

    for det in all_dets:
        x1, y1, x2, y2 = det.bbox
        in_zone = det.track_id in zone_id_set
        alerted = det.track_id in alert_ids

        if alerted:
            color = (0, 0, 255)     # red — alert
            thickness = 3
        elif in_zone:
            color = (0, 255, 100) if det.class_name == "person" else (0, 165, 255)
            thickness = 2
        else:
            color = (60, 60, 60)    # dim grey — outside zone
            thickness = 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if in_zone or alerted:
            label = f"{det.class_name} #{det.track_id} {det.confidence:.0%}"
            cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def run(state=None) -> None:
    """Main pipeline loop.  Blocks until KeyboardInterrupt."""
    cfg_det = _load_yaml("config/detection.yaml")
    cfg_sched = _load_yaml("config/schedule.yaml")
    cfg_alerts = _load_yaml("config/alerts.yaml")

    det_cfg = cfg_det["detector"]
    mask_cfg = cfg_det.get("stationary_mask", {})
    chalk_cfg = cfg_det["chalking"]
    sweep_cfg = cfg_det["sweeper"]

    stream = RTSPHandler(url=os.environ["RTSP_URL"])

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

    initial_polygon = (
        state.zone_polygon if (state and state.zone_polygon)
        else cfg_det["zones"]["street_zone"]["polygon"]
    )
    zone_filter = ZoneFilter(zones={"street_zone": initial_polygon})
    _zone_version = state.get_zone_version() if state else -1

    chalking = ChalkingAnalyzer(
        height_decrease_threshold=chalk_cfg["height_decrease_threshold"],
        history_frames=chalk_cfg["history_frames"],
        cooldown_seconds=chalk_cfg["cooldown_seconds"],
    )

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

    ha_base = os.getenv("HA_WEBHOOK_URL", "").rsplit("/api/", 1)[0]
    notifier = Notifier(
        config=cfg_alerts["alerts"],
        ha_webhook_base=ha_base,
        ha_token=os.getenv("HA_TOKEN", ""),
    )

    stream.start()
    if state:
        state.pipeline_running = True
    logger.info("Pipeline running — press Ctrl-C to stop")

    frame_count = 0
    # track_ids that fired an alert in the current pass (for red bbox overlay)
    alert_ids: set[int] = set()

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                continue

            # Hot-reload zone when updated via the dashboard
            if state:
                v = state.get_zone_version()
                if v != _zone_version:
                    zone_filter = ZoneFilter(zones={"street_zone": state.zone_polygon})
                    _zone_version = v
                    logger.info("Zone reloaded from dashboard")

            frame_count += 1
            all_dets = detector.detect(frame)
            zone_dets = zone_filter.filter(all_dets, "street_zone")

            in_sweep = sweeper.is_sweep_window()
            if state:
                state.sweep_window_active = in_sweep

            active_ids = {d.track_id for d in zone_dets}
            alert_ids.clear()

            for det in zone_dets:
                # ── Chalking ──────────────────────────────────────────────────
                if det.class_name == "person":
                    if chalking.update(det.track_id, det.height):
                        crop = _crop_bytes(frame, det.bbox)
                        result = vlm.analyze(crop)
                        if result["chalking_detected"]:
                            notifier.send("chalking", result, frame, det.bbox)
                            if state:
                                state.record_alert(
                                    "chalking", result["confidence"], result["description"]
                                )
                            alert_ids.add(det.track_id)

                # ── Sweeper ───────────────────────────────────────────────────
                elif det.class_name in {"truck", "motorcycle"} and in_sweep:
                    if sweeper.update(det.track_id, det.center):
                        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result = vlm.analyze(buf.tobytes() if ok else b"")
                        if result["sweeper_detected"]:
                            notifier.send("sweeper", result, frame, det.bbox)
                            if state:
                                state.record_alert(
                                    "sweeper", result["confidence"], result["description"]
                                )
                            alert_ids.add(det.track_id)

            # Evict gone tracks
            for tid in list(chalking._heights.keys()):
                if tid not in active_ids:
                    chalking.evict(tid)
            for tid in list(sweeper._centers.keys()):
                if tid not in active_ids:
                    sweeper.evict(tid)

            # Push annotated frame to web state at capped rate
            if state and frame_count % _PUSH_EVERY_N == 0:
                annotated = _annotate(frame, all_dets, zone_dets, alert_ids)
                state.push_frame(_encode_jpeg(annotated))

    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        stream.stop()
        if state:
            state.pipeline_running = False


def _crop_bytes(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 20) -> bytes:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1 - pad): min(h, y2 + pad), max(0, x1 - pad): min(w, x2 + pad)]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""
