"""Alert dispatcher.

Supports:
  • Home Assistant webhook  (separate IDs per event type so HA automations
                             can distinguish chalking from sweeper)
  • Generic HTTP POST       (any third-party webhook)
  • Snapshot saver          (annotated JPEG written to disk alongside every alert)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        config: dict,
        ha_webhook_base: str = "",
        ha_token: str = "",
    ) -> None:
        self._cfg = config
        self._ha_base = ha_webhook_base.rstrip("/")
        self._ha_token = ha_token
        self._snapshot_dir = Path(config.get("snapshot_dir", "snapshots"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # event_type -> last fire timestamp (monotonic)
        self._last_fire: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def send(
        self,
        event_type: str,        # "chalking" | "sweeper" | "pe_vehicle"
        vlm_result: dict,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        yolo_conf: float | None = None,
    ) -> Path | None:
        """Save a snapshot and fire external alert channels.

        Snapshot is always saved (one per call) so the event log always has a
        thumbnail.  External webhooks (HA, generic) are gated by cooldown and
        confidence threshold.  Returns the snapshot path, or None if snapshot
        saving is disabled.
        """
        # Always save snapshot for the event log thumbnail
        snapshot_path = None
        if self._cfg.get("save_snapshot", True):
            snapshot_path = self._save_snapshot(event_type, frame, bbox, yolo_conf=yolo_conf)

        # Gate external notifications on confidence + cooldown
        if vlm_result.get("confidence", 0.0) < self._cfg.get("min_confidence", 0.70):
            logger.debug("External alert suppressed — confidence below threshold")
            return snapshot_path

        cooldown = self._cfg.get("cooldown_seconds", {}).get(event_type, 300)
        now = time.monotonic()
        if now - self._last_fire.get(event_type, 0) < cooldown:
            logger.debug("External alert suppressed — cooldown active for '%s'", event_type)
            return snapshot_path
        self._last_fire[event_type] = now

        payload = self._build_payload(event_type, vlm_result, snapshot_path)

        if self._cfg.get("home_assistant", {}).get("enabled"):
            threading.Thread(target=self._fire_ha, args=(event_type, payload), daemon=True).start()

        if self._cfg.get("generic_webhook", {}).get("enabled"):
            threading.Thread(target=self._fire_generic, args=(payload,), daemon=True).start()

        logger.info(
            "Alert fired: %s  confidence=%.2f  snapshot=%s",
            event_type,
            vlm_result.get("confidence", 0),
            snapshot_path or "none",
        )
        return snapshot_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_payload(
        self, event_type: str, vlm_result: dict, snapshot_path: Path | None
    ) -> dict:
        return {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": vlm_result.get("confidence"),
            "description": vlm_result.get("description"),
            "snapshot": str(snapshot_path) if snapshot_path else None,
        }

    def _fire_ha(self, event_type: str, payload: dict) -> None:
        ha_cfg = self._cfg["home_assistant"]
        webhook_id_key = f"{event_type}_webhook_id"
        webhook_id = ha_cfg.get(webhook_id_key, f"parking_{event_type}_detected")
        url = f"{self._ha_base}/api/webhook/{webhook_id}"
        headers = {"Authorization": f"Bearer {self._ha_token}"} if self._ha_token else {}
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=5.0)
            resp.raise_for_status()
            logger.debug("HA webhook OK: %s", url)
        except Exception:
            logger.exception("HA webhook failed: %s", url)

    def _fire_generic(self, payload: dict) -> None:
        url = self._cfg["generic_webhook"].get("url", "")
        if not url:
            return
        headers = self._cfg["generic_webhook"].get("headers", {})
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=5.0)
            resp.raise_for_status()
        except Exception:
            logger.exception("Generic webhook failed: %s", url)

    def _save_snapshot(
        self,
        event_type: str,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        yolo_conf: float | None = None,
    ) -> Path:
        annotated = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 0, 255) if event_type == "chalking" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"person {yolo_conf:.2f}" if yolo_conf is not None else event_type.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
            ty = max(y1 - 6, th + 8)
            cv2.rectangle(annotated, (x1, ty - th - 8), (x1 + tw + 8, ty + 2), color, -1)
            cv2.putText(annotated, label, (x1 + 4, ty - 2), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._snapshot_dir / f"{event_type}_{ts}.jpg"
        cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
