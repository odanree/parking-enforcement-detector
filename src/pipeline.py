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
from datetime import datetime
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
from src.stream.osd_timestamp import extract_osd_timestamp
from src.stream.rtsp_handler import RTSPHandler
from src.stream.video_file_handler import VideoFileHandler
from src.vlm.analyzer import VLMAnalyzer

logger = logging.getLogger(__name__)

# ── PE enforcement window (time-of-day gate) ──────────────────────────────────
_PE_WINDOW_ENABLED = os.getenv("PE_WINDOW_ENABLED", "true").lower() == "true"
_PE_WINDOW_START   = int(os.getenv("PE_WINDOW_START", "8"))   # 08:00 local time
_PE_WINDOW_END     = int(os.getenv("PE_WINDOW_END",   "16"))  # 16:00 local time
# Playback mode: VIDEO_PATH set → skip vector-store duplicate suppression so
# replaying the same footage always fires alerts.
_IS_PLAYBACK       = bool(os.getenv("VIDEO_PATH"))


def _in_pe_window(ts: float | None = None) -> bool:
    """True if ts (unix) falls within the PE enforcement window.
    Falls back to datetime.now() when ts is None (live RTSP or unknown source).
    """
    h = datetime.fromtimestamp(ts).hour if ts is not None else datetime.now().hour
    return _PE_WINDOW_START <= h < _PE_WINDOW_END


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


def run(state=None, stream_url: str | None = None, video_path: str | None = None, zone_key: str = "street_zone", vector_store=None, vlm: "VLMAnalyzer | None" = None, confirm_vlm: "VLMAnalyzer | None" = None) -> None:
    """Main pipeline loop.  Blocks until KeyboardInterrupt.

    stream_url:  overrides RTSP_URL env var (RTSP second camera).
    video_path:  overrides VIDEO_PATH env var (file-based second camera).
    zone_key:    which zones entry in detection.yaml to use as the initial polygon.
    """
    cfg_det = _load_yaml("config/detection.yaml")
    cfg_alerts = _load_yaml("config/alerts.yaml")

    det_cfg = cfg_det["detector"]
    mask_cfg = cfg_det.get("stationary_mask", {})
    chalk_cfg   = cfg_det["chalking"]
    _vehicle_proximity_px = int(os.getenv("VEHICLE_PROXIMITY_PX", chalk_cfg.get("vehicle_proximity_px", 250)))
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
        tracker_config=os.getenv("TRACKER_CONFIG", "config/botsort.yaml"),
        fp_grid_cell_px=int(det_cfg.get("fp_grid_cell_px", 48)),
        fp_suppress_seconds=float(os.getenv("FP_SUPPRESS_SECONDS", det_cfg.get("fp_suppress_seconds", 10.0))),
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

    if vlm is None:
        vlm = VLMAnalyzer(
            backend=os.getenv("VLM_BACKEND", "claude"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q4_K_M"),
        )

    # Two-stage pipeline: if CONFIRM_BACKEND is set and different from VLM_BACKEND,
    # positives from the first stage are automatically re-evaluated by the second
    # before an alert fires. Negatives skip the second stage entirely (zero API cost).
    if confirm_vlm is None:
        _confirm_backend = os.getenv("CONFIRM_BACKEND", "")
        if _confirm_backend and _confirm_backend != os.getenv("VLM_BACKEND", "claude"):
            confirm_vlm = VLMAnalyzer(
                backend=_confirm_backend,
                claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
                ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q4_K_M"),
            )
            logger.info("Two-stage VLM: %s → %s (on positive)", os.getenv("VLM_BACKEND"), _confirm_backend)

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
            # Wall time of this frame: NVR timestamp for video files, None for live RTSP
            # (None causes _in_pe_window to fall back to datetime.now())
            _frame_wall_ts = getattr(stream, 'get_current_wall_time', lambda: None)()

            active_ids = {d.track_id for d in zone_dets}
            alert_ids.clear()

            # ── Harvest completed VLM jobs ────────────────────────────────────
            for job_key in list(_vlm_jobs.keys()):
                job = _vlm_jobs[job_key]
                fut, kind, tid, snap_fr, bbox, thumb = job[:6]
                detail_b64 = job[6] if len(job) > 6 else []
                snap_ts    = job[7] if len(job) > 7 else None
                yolo_conf  = job[8] if len(job) > 8 else None
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
                usage          = result.pop("_usage", None)
                rag_neighbors  = result.pop("_rag_neighbors", [])
                pipeline_trace = result.pop("_pipeline_trace", None)
                if pipeline_trace is not None and yolo_conf is not None:
                    pipeline_trace["yolo"] = {
                        "confidence": round(yolo_conf, 3),
                        "track_id":   tid,
                        "bbox":       list(bbox) if bbox else None,
                    }
                if state:
                    state.record_vlm_usage(usage)  # None = Ollama/free, still counts the call
                detected = result.get(f"{kind}_detected", False)
                if state:
                    state.complete_pending_vlm(kind, tid, detected=detected)
                    if not detected:
                        # Full-frame thumbnail (with bbox) for the kanban rejected card
                        _rej_fr = snap_fr.copy()
                        if bbox:
                            x1r, y1r, x2r, y2r = bbox
                            cv2.rectangle(_rej_fr, (x1r, y1r), (x2r, y2r), (0, 0, 255), 2)
                        _, _enc = cv2.imencode('.jpg', _rej_fr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        snap_thumb = _thumb_b64_from_jpeg(_enc.tobytes(), width=640)
                        state.record_rejected_vlm(
                            kind, snap_thumb,
                            result.get("confidence", 0.0),
                            result.get("description", ""),
                            frames=detail_b64 if kind == "chalking" else [],
                            rag_neighbors=rag_neighbors,
                            pipeline_trace=pipeline_trace,
                            track_id=tid,
                        )
                # People-alert phase: record when first-pass VLM was positive.
                # Fires in two-stage mode regardless of confirm outcome so the
                # kanban always shows the full funnel (person near vehicle → confirmed chalking).
                # Single-stage positives land directly in the chalking alert path below.
                _fp = pipeline_trace.get("first_pass") if pipeline_trace else None
                _is_two_stage = pipeline_trace is not None and pipeline_trace.get("confirm") is not None
                if kind == "chalking" and _fp and _fp.get("detected") and _is_two_stage and state:
                    _pa_ts = snap_ts or extract_osd_timestamp(snap_fr)
                    state.record_people_alert(
                        confidence=_fp.get("confidence", 0.0),
                        description=_fp.get("description", ""),
                        frames=detail_b64,
                        timestamp=_pa_ts,
                        track_id=tid,
                        rag_neighbors=rag_neighbors,
                        pipeline_trace=pipeline_trace,
                        thumbnail=thumb,
                    )

                _is_duplicate = False
                if kind == "chalking" and vector_store:
                    try:
                        _event_id = vector_store.add(
                            description=result.get("description", ""),
                            detected=detected,
                            confidence=result.get("confidence", 0.0),
                            camera_id=state.camera_id if state else 0,
                            thumbnail_b64=thumb,
                            frames_b64=detail_b64,
                        )
                        _is_duplicate = (not _IS_PLAYBACK) and (_event_id == "")
                    except Exception:
                        logger.exception("vector_store.add failed")
                if kind == "chalking" and result["chalking_detected"] and not _is_duplicate:
                    sf = _apply_privacy(snap_fr, state.privacy_regions) if (state and state.privacy_mode) else snap_fr
                    snap = notifier.send("chalking", result, sf, bbox)
                    event_ts = snap_ts or extract_osd_timestamp(snap_fr)
                    if state:
                        state.record_alert("chalking", result["confidence"], result["description"], snapshot=snap.name if snap else None, frames=detail_b64, track_id=tid, timestamp=event_ts, rag_neighbors=rag_neighbors, pipeline_trace=pipeline_trace)
                    alert_ids.add(tid)
                    chalking.on_alert(tid)
                elif kind == "chalking" and not result["chalking_detected"]:
                    # High-confidence rejection — put track on cooldown so the same
                    # pedestrian doesn't re-trigger VLM calls every sample_every_n frames.
                    if result.get("confidence", 0.0) >= 0.80:
                        chalking.on_alert(tid)

            vehicle_bboxes = [d.bbox for d in all_dets if d.class_name in {"car", "truck", "motorcycle"}]

            for det in zone_dets:
                # ── Chalking ──────────────────────────────────────────────────
                if det.class_name == "person":
                    if not _near_vehicle(det.bbox, vehicle_bboxes, _vehicle_proximity_px):
                        chalking.evict(det.track_id)  # reset state — pedestrian, not near any vehicle
                        continue
                    crop = _crop_wide_bytes(frame, det.bbox)
                    chalking.store_frame(det.track_id, crop)
                    if chalking.update(det.track_id):
                        job_key = ("chalking", det.track_id)
                        if job_key not in _vlm_jobs:
                            detail_frames = chalking.get_frame_buffer(det.track_id) or [crop]
                            thumb      = _thumb_b64_from_jpeg(detail_frames[-1])
                            detail_b64 = [_thumb_b64_from_jpeg(f) for f in detail_frames]

                            suppressed_sid = state.find_open_suppressed_session() if state else None
                            trusted_sid    = state.find_open_trusted_session()    if state else None

                            if suppressed_sid:
                                # Skip VLM entirely — record in debug log and set cooldown
                                if state:
                                    state.record_rejected_vlm(
                                        "chalking", thumb, 0.0,
                                        f"Suppressed — session {suppressed_sid} downvoted by user",
                                        suppressed_by_session=suppressed_sid,
                                    )
                                chalking.on_alert(det.track_id)
                            elif trusted_sid:
                                # Auto-confirm — fire alert without a VLM call
                                auto_desc = f"Auto-confirmed — session {trusted_sid} upvoted by user"
                                sf = _apply_privacy(frame, state.privacy_regions) if (state and state.privacy_mode) else frame
                                snap = notifier.send("chalking", {"chalking_detected": True, "confidence": 1.0, "description": auto_desc}, sf, det.bbox)
                                osd_ts = extract_osd_timestamp(frame)
                                if state:
                                    state.record_alert("chalking", 1.0, auto_desc, snapshot=snap.name if snap else None, frames=detail_b64, track_id=det.track_id, timestamp=osd_ts)
                                alert_ids.add(det.track_id)
                                chalking.on_alert(det.track_id)
                            else:
                                # After-hours low-confidence gate: skip VLM when outside the
                                # PE window and YOLO confidence is below 25% (not in playback).
                                _after_hours = not _IS_PLAYBACK and not _in_pe_window(_frame_wall_ts)
                                if _after_hours and det.confidence < 0.40:
                                    if state:
                                        state.record_rejected_vlm(
                                            "chalking", thumb, det.confidence,
                                            f"After-hours gate: YOLO {det.confidence:.0%} below 40% threshold",
                                            rejection_reason="out_of_hours",
                                            track_id=det.track_id,
                                            bbox=list(det.bbox) if det.bbox else None,
                                        )
                                    chalking.on_alert(det.track_id)
                                    continue
                                scene      = _crop_scene_bytes(frame, det.bbox)
                                vlm_frames = [scene] + detail_frames
                                snap_ts    = _frame_wall_ts
                                # confirm_vlm intentionally not passed — Claude is a human-triggered gate only.
                                # Use /api/dataset/reeval to run Claude on specific items.
                                fut = _vlm_pool.submit(_two_stage, vlm, None, vlm_frames, "chalking", vector_store)
                                _vlm_jobs[job_key] = (fut, "chalking", det.track_id, frame.copy(), det.bbox, thumb, detail_b64, snap_ts, det.confidence)
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


_RAG_AUTO_REJECT   = os.getenv("RAG_AUTO_REJECT", "false").lower() == "true"
_RAG_FP_THRESHOLD  = float(os.getenv("RAG_FP_THRESHOLD", "0.30"))
_RAG_FP_MIN_VOTES  = int(os.getenv("RAG_FP_MIN_VOTES", "3"))


def _two_stage(
    primary: "VLMAnalyzer",
    confirm: "VLMAnalyzer | None",
    frames: "list[bytes]",
    kind: str,
    rag_store=None,
) -> dict:
    """Run primary VLM; RAG lookup; optionally skip confirm; run confirm VLM.

    Returns the final result dict plus internal keys:
      _usage          — token usage from whichever VLM ran last
      _rag_neighbors  — list of similar labeled events from ChromaDB
      _pipeline_trace — per-stage breakdown for the kanban view
    """
    first = primary.analyze(frames, kind)

    # ── Stage 1 snapshot ─────────────────────────────────────────────────────
    trace: dict = {
        "first_pass": {
            "detected":    first.get(f"{kind}_detected", False),
            "confidence":  first.get("confidence", 0.0),
            "description": first.get("description", ""),
            "backend":     primary._backend,
        },
        "rag": None,
        "confirm": None,
    }

    # ── RAG lookup ────────────────────────────────────────────────────────────
    rag_neighbors: list[dict] = []
    if rag_store is not None and first.get("description"):
        try:
            raw = rag_store.query_similar_by_text(first["description"], n=5)
            rag_neighbors = [
                {
                    "id":          n["id"],
                    "label":       n.get("label", ""),
                    "distance":    n["distance"],
                    "description": n["description"][:120],
                    "detected":    int(n.get("detected", 0)),
                }
                for n in raw
            ]
        except Exception:
            logger.debug("RAG lookup failed", exc_info=True)

    labeled    = [n for n in rag_neighbors if n["label"]]
    fp_close   = [n for n in labeled if n["label"] == "false_positive" and n["distance"] < _RAG_FP_THRESHOLD]
    tp_close   = [n for n in labeled if n["label"] == "true_positive"  and n["distance"] < _RAG_FP_THRESHOLD]
    rag_signal = "fp" if fp_close else ("tp" if tp_close else "none")

    trace["rag"] = {
        "neighbors":      rag_neighbors,
        "labeled_count":  len(labeled),
        "fp_close":       len(fp_close),
        "tp_close":       len(tp_close),
        "signal":         rag_signal,
        "skipped_confirm": False,
    }

    # ── RAG auto-reject (opt-in via RAG_AUTO_REJECT=true) ────────────────────
    if (
        _RAG_AUTO_REJECT
        and confirm is not None
        and first.get(f"{kind}_detected", False)
        and len(fp_close) >= _RAG_FP_MIN_VOTES
    ):
        first[f"{kind}_detected"] = False
        first["description"] += f" [RAG: auto-rejected — {len(fp_close)} close FP matches]"
        trace["rag"]["skipped_confirm"] = True
        first["_rag_neighbors"]  = rag_neighbors
        first["_pipeline_trace"] = trace
        logger.info("RAG auto-reject: %d FP neighbors (dist<%.2f)", len(fp_close), _RAG_FP_THRESHOLD)
        return first

    first["_rag_neighbors"]  = rag_neighbors
    first["_pipeline_trace"] = trace

    if confirm is None or not first.get(f"{kind}_detected", False):
        return first

    # ── Confirm stage ─────────────────────────────────────────────────────────
    logger.debug("Two-stage: %s positive → confirm VLM", kind)
    second = confirm.analyze(frames, kind)
    trace["confirm"] = {
        "detected":    second.get(f"{kind}_detected", False),
        "confidence":  second.get("confidence", 0.0),
        "description": second.get("description", ""),
        "backend":     confirm._backend,
    }
    second["_rag_neighbors"]  = rag_neighbors
    second["_pipeline_trace"] = trace
    return second


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


def _near_vehicle(
    person_bbox: tuple[int, int, int, int],
    vehicle_bboxes: list[tuple[int, int, int, int]],
    threshold_px: int,
) -> bool:
    """Return True if the person bbox is within threshold_px of any vehicle bbox."""
    px1, py1, px2, py2 = person_bbox
    for vx1, vy1, vx2, vy2 in vehicle_bboxes:
        dx = max(0, max(px1, vx1) - min(px2, vx2))
        dy = max(0, max(py1, vy1) - min(py2, vy2))
        if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
            return True
    return False
