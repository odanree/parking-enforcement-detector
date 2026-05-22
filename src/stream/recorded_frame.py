"""Fetch a hi-res frame from NVR *recorded* footage at a past timestamp.

Powers the "capture hi-res" button on historical dataset events: a low-res
zone_pedestrian (or any) event stored only a thumbnail, but the NVR still has
the full recording on disk (within its retention window). Given the event's
timestamp + camera, this seeks the recording and grabs a native-resolution
frame.

Two strategies, tried in order:
  1. Dahua/Amcrest time-based playback RTSP — starts playback at the exact
     requested time, so the first decoded frame is ~the event moment.
  2. Fallback: locate the recording file via RPC2 mediaFileFind
     (amcrest_api.find_recording_rtsp) and read into it.

Returns native JPEG bytes, or None if disabled, unmapped, or the footage is
out of retention / unreachable.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import cv2

from src.stream.amcrest_api import find_recording_rtsp

logger = logging.getLogger(__name__)

# Force TCP for playback RTSP (same rationale as live — UDP smears frames).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_RECORDED_ENABLED = os.getenv("RECORDED_FRAME_ENABLED", "true").lower() == "true"


def _nvr_host() -> str:
    return os.getenv("AMCREST_HOST", "")


def _nvr_creds() -> tuple[str, str]:
    return os.getenv("AMCREST_USER", "admin"), os.getenv("AMCREST_PASS", "")


def _nvr_channel(camera_id: int) -> int:
    # NVR channel for this camera (1-indexed), shared with PTZ / snapshot config.
    return int(os.getenv(f"PTZ_CHANNEL_{camera_id}", str(camera_id + 1)))


def _nvr_port() -> str:
    return os.getenv("AMCREST_PORT", "554")


def _grab_first_good(url: str, settle_frames: int, timeout_ms: int = 12000) -> bytes | None:
    """Open an RTSP URL, skip a few frames to let the decoder settle, return a
    native-res JPEG of the next frame."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(max(1, settle_frames)):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


def fetch_recorded_frame(camera_id: int, ts_unix: float, settle_frames: int = 8) -> bytes | None:
    """Return a native-res JPEG from the recording at ts_unix for camera_id, or None."""
    if not _RECORDED_ENABLED:
        return None
    host = _nvr_host()
    if not host:
        logger.warning("recorded_frame: AMCREST_HOST not set")
        return None
    user, pwd = _nvr_creds()
    ch = _nvr_channel(camera_id)
    port = _nvr_port()
    dt = datetime.fromtimestamp(ts_unix)   # container TZ = NVR local time

    # ── Strategy 1: time-based playback RTSP (precise start) ─────────────────
    start = dt.strftime("%Y_%m_%d_%H_%M_%S")
    end = (dt + timedelta(seconds=15)).strftime("%Y_%m_%d_%H_%M_%S")
    play_url = (
        f"rtsp://{user}:{pwd}@{host}:{port}/cam/playback"
        f"?channel={ch}&starttime={start}&endtime={end}"
    )
    jpeg = _grab_first_good(play_url, settle_frames)
    if jpeg:
        logger.info("recorded_frame: time-playback OK cam=%d ch=%d ts=%s (%d bytes)",
                    camera_id, ch, start, len(jpeg))
        return jpeg
    logger.info("recorded_frame: time-playback miss cam=%d ch=%d ts=%s — trying file lookup",
                camera_id, ch, start)

    # ── Strategy 2: locate the recording file, read into it ──────────────────
    pre_roll = 4
    file_url = find_recording_rtsp(host, user, pwd, port, ch, dt, pre_roll_seconds=pre_roll)
    if not file_url:
        logger.warning("recorded_frame: no recording found cam=%d ch=%d ts=%s (out of retention?)",
                        camera_id, ch, start)
        return None
    # File playback starts at the file's beginning; read ~pre_roll seconds of
    # frames (assume ~15 fps) to approach the event moment, then grab.
    jpeg = _grab_first_good(file_url, settle_frames + pre_roll * 15)
    if jpeg:
        logger.info("recorded_frame: file-playback OK cam=%d ch=%d (%d bytes)", camera_id, ch, len(jpeg))
    return jpeg
