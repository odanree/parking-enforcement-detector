"""Query Amcrest/Dahua NVR via JSON-RPC2 API to find recording file paths.

The CGI endpoint (/cgi-bin/mediaFileFind.cgi) returns 400 on many firmware
versions. The RPC2 JSON-RPC protocol at /RPC2_Login + /RPC2 is the canonical
Dahua/Amcrest API and works across firmware versions.

Auth flow:
  1. POST /RPC2_Login with empty password → NVR returns realm + random nonce
  2. MD5(user:realm:password) → type1
     MD5(user:random:type1)   → final hash
  3. POST /RPC2_Login with hash → session token
  4. POST /RPC2 mediaFileFind.factory.instance → object ID
  5. POST /RPC2 mediaFileFind.findFile → True/False
  6. POST /RPC2 mediaFileFind.findNextFile → list of file infos with FilePath
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)


def _rpc2_login(client: httpx.Client, host: str, user: str, pwd: str) -> str | None:
    """Perform two-step RPC2 login. Returns session token or None."""
    # Step 1: challenge
    r1 = client.post(f"http://{host}/RPC2_Login", json={
        "method": "global.login",
        "params": {
            "userName": user, "password": "",
            "clientType": "Web3.0",
            "authorityType": "Default", "passwordType": "Default",
        },
        "id": 1, "session": "",
    })
    if r1.status_code != 200:
        logger.warning("RPC2 login challenge failed: %d %s", r1.status_code, r1.text[:200])
        return None

    d1 = r1.json()
    session  = d1.get("session", "")
    realm    = d1.get("params", {}).get("realm", "")
    random_s = d1.get("params", {}).get("random", "")
    logger.debug("RPC2 challenge session=%s realm=%s random=%s", session, realm, random_s)

    # Step 2: MD5 hash and authenticate
    type1  = hashlib.md5(f"{user}:{realm}:{pwd}".encode()).hexdigest().upper()
    pwd_md5 = hashlib.md5(f"{user}:{random_s}:{type1}".encode()).hexdigest().upper()

    r2 = client.post(f"http://{host}/RPC2_Login", json={
        "method": "global.login",
        "params": {
            "userName": user, "password": pwd_md5,
            "clientType": "Web3.0",
            "authorityType": "Default", "passwordType": "MD5",
        },
        "id": 2, "session": session,
    })
    if r2.status_code != 200:
        logger.warning("RPC2 auth failed: %d %s", r2.status_code, r2.text[:200])
        return None

    d2 = r2.json()
    logger.debug("RPC2 auth result: %s", d2)
    if not d2.get("result"):
        logger.warning("RPC2 login rejected: %s", d2)
        return None

    return d2.get("session", session)


def find_recording_rtsp(
    host: str,
    user: str,
    pwd: str,
    rtsp_port: str | int,
    nvr_channel: int,          # 1-indexed (same as RTSP channel= param)
    dt: datetime,
    pre_roll_seconds: int = 30,
) -> str | None:
    """Return an RTSP URL for the specific NVR recording at dt, or None."""
    start = dt - timedelta(seconds=pre_roll_seconds)
    end   = start + timedelta(hours=2)
    start_s = start.strftime("%Y-%m-%d %H:%M:%S")
    end_s   = end.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with httpx.Client(timeout=10) as client:
            session = _rpc2_login(client, host, user, pwd)
            if session is None:
                return None

            # Create mediaFileFind instance
            r3 = client.post(f"http://{host}/RPC2", json={
                "method": "mediaFileFind.factory.instance",
                "params": None, "id": 3, "session": session,
            })
            d3 = r3.json()
            object_id = d3.get("result")
            logger.debug("factory.instance → %s", d3)
            if object_id is None:
                logger.warning("RPC2 factory.instance returned no object ID: %s", d3)
                return None

            # Try 0-indexed then 1-indexed channel
            for cgi_ch in [nvr_channel - 1, nvr_channel]:
                r4 = client.post(f"http://{host}/RPC2", json={
                    "method": "mediaFileFind.findFile",
                    "params": {
                        "object": object_id,
                        "count": 4,
                        "condition": {
                            "Channel": cgi_ch,
                            "StartTime": start_s,
                            "EndTime": end_s,
                        },
                    },
                    "id": 4, "session": session,
                })
                d4 = r4.json()
                logger.info("RPC2 findFile ch=%d result=%s", cgi_ch, d4.get("result"))

                if not d4.get("result"):
                    continue

                r5 = client.post(f"http://{host}/RPC2", json={
                    "method": "mediaFileFind.findNextFile",
                    "params": {"object": object_id, "count": 4},
                    "id": 5, "session": session,
                })
                d5 = r5.json()
                logger.info("RPC2 findNextFile ch=%d: %s", cgi_ch, str(d5)[:400])

                infos = d5.get("params", {}).get("infos", [])
                if not infos:
                    continue

                file_path = infos[0].get("FilePath", "")
                if not file_path:
                    continue

                rtsp_url = f"rtsp://{user}:{pwd}@{host}:{rtsp_port}{file_path}"
                safe_url = f"rtsp://****:****@{host}:{rtsp_port}{file_path}"
                logger.info("Amcrest recording → %s", safe_url)
                return rtsp_url

    except Exception as exc:
        logger.warning("Amcrest RPC2 failed: %s", exc)

    return None
