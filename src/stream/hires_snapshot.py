"""Fetch a full-resolution JPEG snapshot from Amcrest/Dahua camera HTTP API.

Uses the CGI snapshot endpoint which returns a native-resolution JPEG without
requiring a second continuous RTSP stream.

  GET /cgi-bin/snapshot.cgi?channel=<ch>&subtype=0   (main stream — highest res)
  GET /cgi-bin/snapshot.cgi?channel=<ch>&subtype=1   (sub stream — lower res)

Environment variables (shared with ptz.py):
  AMCREST_HOST        NVR/camera IP (fallback when no per-camera override)
  AMCREST_USER        HTTP auth username
  AMCREST_PASS        HTTP auth password
  PTZ_HOST_{n}        Per-camera host override (shared with PTZ)
  PTZ_CHANNEL_{n}     NVR channel for camera n (shared with PTZ)
  PTZ_PORT            HTTP port (default: 80)
  HIRES_SNAPSHOT      Set to "true" to enable hi-res capture (default: false)
  HIRES_SUBTYPE       0 = main stream, 1 = sub stream (default: 0)

Per-camera snapshot overrides (take priority over PTZ_* / AMCREST_* vars):
  SNAPSHOT_HOST_{n}   Direct camera IP — bypasses NVR for the snapshot fetch
  SNAPSHOT_USER_{n}   Camera username  (defaults to PTZ_USER_{n} / AMCREST_USER)
  SNAPSHOT_PASS_{n}   Camera password  (defaults to PTZ_PASS_{n} / AMCREST_PASS)
  SNAPSHOT_CHANNEL_{n} Channel on the direct camera (default: 1)
  SNAPSHOT_PORT       HTTP port for direct camera (default: same as PTZ_PORT)
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_ENABLED  = os.getenv("HIRES_SNAPSHOT", "false").lower() == "true"
_SUBTYPE  = int(os.getenv("HIRES_SUBTYPE", "0"))
_PTZ_PORT = os.getenv("PTZ_PORT", "80")
_SNAP_PORT = os.getenv("SNAPSHOT_PORT", _PTZ_PORT)


def _snap_host(camera_id: int) -> str:
    """Direct-camera host if configured, otherwise NVR host."""
    return (
        os.getenv(f"SNAPSHOT_HOST_{camera_id}")
        or os.getenv(f"PTZ_HOST_{camera_id}")
        or os.getenv("AMCREST_HOST", "")
    )


def _snap_creds(camera_id: int) -> tuple[str, str]:
    user = (
        os.getenv(f"SNAPSHOT_USER_{camera_id}")
        or os.getenv(f"PTZ_USER_{camera_id}")
        or os.getenv("AMCREST_USER", "admin")
    )
    pwd = (
        os.getenv(f"SNAPSHOT_PASS_{camera_id}")
        or os.getenv(f"PTZ_PASS_{camera_id}")
        or os.getenv("AMCREST_PASS", "")
    )
    return user, pwd


def _snap_channel(camera_id: int) -> int:
    """Channel to request.  Defaults to 1 when using a direct-camera host
    (standalone cameras are always channel 1), or the NVR channel otherwise."""
    explicit = os.getenv(f"SNAPSHOT_CHANNEL_{camera_id}")
    if explicit:
        return int(explicit)
    if os.getenv(f"SNAPSHOT_HOST_{camera_id}"):
        return 1  # direct camera — always channel 1
    return int(os.getenv(f"PTZ_CHANNEL_{camera_id}", str(camera_id + 1)))


def _snap_port(camera_id: int) -> str:
    return os.getenv(f"SNAPSHOT_PORT_{camera_id}", _SNAP_PORT)


def fetch_hires_jpeg(camera_id: int, subtype: int | None = None) -> bytes | None:
    """Fetch a hi-res JPEG from the camera's HTTP snapshot API.

    Returns JPEG bytes on success, None if disabled or on any error.
    """
    if not _ENABLED:
        return None

    host = _snap_host(camera_id)
    if not host:
        return None

    user, pwd = _snap_creds(camera_id)
    ch   = _snap_channel(camera_id)
    sub  = subtype if subtype is not None else _SUBTYPE
    port = _snap_port(camera_id)
    url  = f"http://{host}:{port}/cgi-bin/snapshot.cgi"
    direct = bool(os.getenv(f"SNAPSHOT_HOST_{camera_id}"))

    try:
        with httpx.Client(auth=httpx.DigestAuth(user, pwd), timeout=5.0) as c:
            r = c.get(url, params={"channel": ch, "subtype": sub})
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "image" not in ct:
                logger.warning(
                    "hires_snapshot: unexpected content-type %r cam=%d url=%s",
                    ct, camera_id, url,
                )
                return None
            logger.info(
                "hires_snapshot: OK %d bytes cam=%d ch=%d subtype=%d source=%s",
                len(r.content), camera_id, ch, sub,
                "direct" if direct else "nvr",
            )
            return r.content
    except Exception:
        logger.warning("hires_snapshot failed cam=%d url=%s", camera_id, url, exc_info=True)
        return None
