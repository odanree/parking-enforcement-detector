"""Detection pipeline — runs in the main thread (headless) or a daemon thread
(when the web dashboard is active).

Call `run(state)` to start.  Pass an `AppState` instance so the web layer
can read annotated frames and events in real-time.  Pass `None` for headless.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.alerts.notifier import Notifier
from src.behavior.chalking_analyzer import ChalkingAnalyzer
from src.detection.motion_detector import MotionDetector
from src.detection.object_detector import Detection, ObjectDetector
from src.detection.zone_filter import ZoneFilter
from src.stream.frame_undistorter import FrameUndistorter
from src.stream.rtsp_handler import RTSPHandler
from src.stream.video_file_handler import VideoFileHandler
from src.vlm.analyzer import VLMAnalyzer

logger = logging.getLogger(__name__)

# Push at most this many frames per second to the WebSocket state.
_MAX_PUSH_FPS = 20
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
            source = "MOG2" if det.track_id >= 1000 else "YOLO"
            label = f"[{source}] {det.class_name} #{det.track_id} {det.confidence:.0%}"
            cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def run(state=None, stream_url: str | None = None, video_path: str | None = None, zone_key: str = "street_zone", vector_store=None) -> None:
    """Main pipeline loop.  Blocks until KeyboardInterrupt.

    stream_url:  overrides RTSP_URL env var (RTSP second camera).
    video_path:  overrides VIDEO_PATH env var (file-based second camera).
    zone_key:    which zones entry in detection.yaml to use as the initial polygon.
    """
    cfg_det = _load_yaml("config/detection.yaml")
    cfg_alerts = _load_yaml("config/alerts.yaml")

    det_cfg = cfg_det["detector"]
    mask_cfg = cfg_det.get("stationary_mask", {})
    chalk_cfg = cfg_det["chalking"]
    mot_cfg = cfg_det.get("motion_detector", {})

    video_path = video_path or (os.getenv("VIDEO_PATH", "") if stream_url is None else "")
    if video_path:
        stream = VideoFileHandler(
            path=video_path,
            loop=os.getenv("VIDEO_LOOP", "true").lower() != "false",
            speed=float(os.getenv("VIDEO_SPEED", "1.0")),
        )
    else:
        url = stream_url or os.environ["RTSP_URL"]
        stream = RTSPHandler(url=url)

    _det_w = int(os.getenv("INPUT_WIDTH", det_cfg["input_width"]))
    _det_h = int(os.getenv("INPUT_HEIGHT", det_cfg["input_height"]))

    detector = ObjectDetector(
        model_path=os.getenv("YOLO_MODEL", det_cfg["model"]),
        threshold=float(os.getenv("INFERENCE_THRESHOLD", det_cfg["threshold"])),
        input_size=(_det_w, _det_h),
        min_area_fraction=det_cfg["min_area_fraction"],
        max_area_fraction=det_cfg["max_area_fraction"],
        stationary_px=mask_cfg.get("pixel_threshold", 15),
        stationary_frames=mask_cfg.get("frames", 30),
    )

    initial_polygon = (
        state.zone_polygon if (state and state.zone_polygon)
        else cfg_det["zones"][zone_key]["polygon"]
    )
    zone_filter = ZoneFilter(zones={"street_zone": initial_polygon})
    _zone_version = state.get_zone_version() if state else -1

    chalking = ChalkingAnalyzer(
        entry_frames=chalk_cfg.get("entry_frames", 10),
        sample_every_n=chalk_cfg.get("sample_every_n", 30),
        cooldown_seconds=chalk_cfg["cooldown_seconds"],
        frame_buffer_size=chalk_cfg.get("frame_buffer_size", 4),
        buffer_sample_every_n=chalk_cfg.get("buffer_sample_every_n", 5),
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

    motion_detector = MotionDetector(
        history=mot_cfg.get("history", 500),
        var_threshold=float(mot_cfg.get("var_threshold", 25)),
        min_area=int(mot_cfg.get("min_area", 600)),
        max_area=int(mot_cfg.get("max_area", 12000)),
        edge_margin=int(mot_cfg.get("edge_margin", 80)),
        seam_x=int(mot_cfg.get("seam_x", 0)),
        seam_margin=int(mot_cfg.get("seam_margin", 0)),
    )

    undistorter: FrameUndistorter | None = None
    if os.getenv("UNDISTORT_FRAME", "false").lower() == "true":
        undistorter = FrameUndistorter("config/camera_calibration.yaml")

    _vlm_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm")
    # Each entry: (future, kind, track_id, snap_frame, bbox, thumb_b64)
    _vlm_jobs: dict[tuple[str, int], tuple[Future, str, int, np.ndarray, tuple, str]] = {}


    _initial_paused = os.getenv("INITIAL_PAUSED", "false").lower() == "true"
    if video_path and _initial_paused:
        stream.pause()

    stream.start()
    if state:
        state.set_stream(stream)
        state.pipeline_running = True
        if video_path and _initial_paused:
            state.paused = True
    logger.info("Pipeline running — press Ctrl-C to stop")

    frame_count = 0
    _last_jpeg: bytes = b""
    # track_ids that fired an alert in the current pass (for red bbox overlay)
    alert_ids: set[int] = set()

    try:
        while True:
            # ── Pause gate ────────────────────────────────────────────────────
            if state and state.paused:
                if _last_jpeg and state:
                    state.push_frame(_last_jpeg)
                time.sleep(0.033)
                continue

            frame = stream.get_frame()
            if frame is None:
                continue

            if undistorter is not None:
                frame = undistorter.undistort(frame)

            # Normalize to detection resolution so bbox coords match the
            # annotated frame. NVR footage is often higher-res than 1280×720.
            fh, fw = frame.shape[:2]
            if (fw, fh) != (_det_w, _det_h):
                frame = cv2.resize(frame, (_det_w, _det_h))

            # Hot-reload zone when updated via the dashboard
            if state:
                v = state.get_zone_version()
                if v != _zone_version:
                    zone_filter = ZoneFilter(zones={"street_zone": state.zone_polygon})
                    _zone_version = v
                    logger.info("Zone reloaded from dashboard")

            frame_count += 1
            if state:
                state.tick_fps()
            all_dets = detector.detect(frame)

            # Motion-detect mode: supplement YOLO with MOG2 blobs for missed persons.
            # YOLO persons are always kept — replacing them with MOG2 caused persons
            # walking in front of vehicles to vanish (blob overlapped car bbox → filtered).
            if state and state.motion_detect_enabled:
                motion_dets = motion_detector.detect(frame)
                vehicle_bboxes = [
                    d.bbox for d in all_dets
                    if d.class_name in {"car", "truck", "motorcycle"}
                ]
                yolo_person_bboxes = [
                    d.bbox for d in all_dets if d.class_name == "person"
                ]
                motion_dets = [
                    d for d in motion_dets
                    if not _overlaps_vehicle(d.bbox, vehicle_bboxes)
                    and not _overlaps_vehicle(d.bbox, yolo_person_bboxes, min_overlap_frac=0.30)
                ]
                all_dets = all_dets + motion_dets

            zone_dets = zone_filter.filter(all_dets, "street_zone")

            active_ids = {d.track_id for d in zone_dets}
            alert_ids.clear()

            # ── Harvest completed VLM jobs ────────────────────────────────────
            for job_key in list(_vlm_jobs.keys()):
                job = _vlm_jobs[job_key]
                fut, kind, tid, snap_fr, bbox, thumb = job[:6]
                detail_b64 = job[6] if len(job) > 6 else []
                if not fut.done():
                    continue
                del _vlm_jobs[job_key]
                try:
                    result = fut.result()
                except Exception:
                    logger.exception("VLM job (%s #%d) raised", kind, tid)
                    if state:
                        state.complete_pending_vlm(kind, tid, detected=False)
                    continue
                detected = result.get(f"{kind}_detected", False)
                if state:
                    state.complete_pending_vlm(kind, tid, detected=detected)
                    if not detected:
                        state.record_rejected_vlm(
                            kind, thumb,
                            result.get("confidence", 0.0),
                            result.get("description", ""),
                            frames=detail_b64 if kind == "chalking" else [],
                        )
                if kind == "chalking" and vector_store:
                    try:
                        vector_store.add(
                            description=result.get("description", ""),
                            detected=detected,
                            confidence=result.get("confidence", 0.0),
                            camera_id=state.camera_id if state else 0,
                            thumbnail_b64=thumb,
                            frames_b64=detail_b64,
                        )
                    except Exception:
                        logger.exception("vector_store.add failed")
                if kind == "chalking" and result["chalking_detected"]:
                    sf = _apply_privacy(snap_fr, state.privacy_regions) if (state and state.privacy_mode) else snap_fr
                    snap = notifier.send("chalking", result, sf, bbox)
                    if state:
                        state.record_alert("chalking", result["confidence"], result["description"], snapshot=snap.name if snap else None, frames=detail_b64)
                    alert_ids.add(tid)
                    chalking.on_alert(tid)

            for det in zone_dets:
                # ── Chalking ──────────────────────────────────────────────────
                if det.class_name == "person":
                    crop = _crop_wide_bytes(frame, det.bbox)
                    chalking.store_frame(det.track_id, crop)
                    if chalking.update(det.track_id):
                        job_key = ("chalking", det.track_id)
                        if job_key not in _vlm_jobs:
                            detail_frames = chalking.get_frame_buffer(det.track_id) or [crop]
                            scene         = _crop_scene_bytes(frame, det.bbox)
                            vlm_frames    = [scene] + detail_frames  # scene first for vehicle-state check
                            thumb  = _thumb_b64_from_jpeg(detail_frames[-1])
                            detail_b64 = [_thumb_b64_from_jpeg(f) for f in detail_frames]
                            fut = _vlm_pool.submit(vlm.analyze, vlm_frames, "chalking")
                            _vlm_jobs[job_key] = (fut, "chalking", det.track_id, frame.copy(), det.bbox, thumb, detail_b64)
                            if state:
                                state.add_pending_vlm("chalking", det.track_id, thumb)

            # Evict gone tracks
            for tid in list(chalking._frame_count.keys()):
                if tid not in active_ids:
                    chalking.evict(tid)

            # Push annotated frame to web state at capped rate
            if state and frame_count % _PUSH_EVERY_N == 0:
                annotated = _annotate(frame, all_dets, zone_dets, alert_ids)
                if state.privacy_mode and state.privacy_regions:
                    annotated = _apply_privacy(annotated, state.privacy_regions)
                _last_jpeg = _encode_jpeg(annotated)
                state.push_frame(_last_jpeg)

    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        _vlm_pool.shutdown(wait=False)
        stream.stop()
        if state:
            state.pipeline_running = False


def _overlaps_vehicle(
    motion_bbox: tuple[int, int, int, int],
    vehicle_bboxes: list[tuple[int, int, int, int]],
    pad: int = 5,
    min_overlap_frac: float = 0.50,
) -> bool:
    """Return True if the motion blob is mostly inside a vehicle bbox.

    Minimal padding (5px) so the gap between parked cars isn't accidentally
    covered.  Requires 50% overlap so a person standing next to a car isn't
    rejected just because their blob grazes the car bbox edge.
    """
    mx1, my1, mx2, my2 = motion_bbox
    blob_area = max(1, (mx2 - mx1) * (my2 - my1))
    for x1, y1, x2, y2 in vehicle_bboxes:
        ix1 = max(mx1, x1 - pad)
        iy1 = max(my1, y1 - pad)
        ix2 = min(mx2, x2 + pad)
        iy2 = min(my2, y2 + pad)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / blob_area >= min_overlap_frac:
                return True
    return False


def _crop_bytes(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 20) -> bytes:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1 - pad): min(h, y2 + pad), max(0, x1 - pad): min(w, x2 + pad)]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _apply_privacy(frame: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    """Black out privacy regions (license plates, etc.) in-place on a copy."""
    if not regions:
        return frame
    out = frame.copy()
    fh, fw = out.shape[:2]
    for x1, y1, x2, y2 in regions:
        out[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)] = 0
    return out


def _thumb_b64_from_jpeg(jpeg_bytes: bytes, width: int = 200) -> str:
    """Resize a JPEG to a small thumbnail for the pending-jobs UI card."""
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    height = max(1, int(h * width / w))
    thumb = cv2.resize(img, (width, height))
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _crop_tight_bytes(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Tight crop for debug thumbnails — person fills most of the image."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(bw, 30)
    pad_y = max(bh, 30)
    crop = frame[max(0, y1 - pad_y) : min(h, y2 + pad_y),
                 max(0, x1 - pad_x) : min(w, x2 + pad_x)]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _crop_wide_bytes(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Close-up crop for per-frame person detail (tool, posture)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x     = max(bw * 3, 120)
    pad_above = max(bh * 2,  60)
    pad_below = max(bh,      40)
    crop = frame[max(0, y1 - pad_above) : min(h, y2 + pad_below),
                 max(0, x1 - pad_x)     : min(w, x2 + pad_x)]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else b""


def _crop_scene_bytes(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Wide scene crop showing all nearby vehicles — used as the first frame sent
    to the VLM so it can assess vehicle state (open trunks, doors) before posture."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x     = max(bw * 10, 350)   # wide enough to capture full adjacent vehicles
    pad_above = max(bh *  4, 150)
    pad_below = max(bh *  4, 150)
    crop = frame[max(0, y1 - pad_above) : min(h, y2 + pad_below),
                 max(0, x1 - pad_x)     : min(w, x2 + pad_x)]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""
