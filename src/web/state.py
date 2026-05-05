"""Shared in-process state between the detection pipeline and the web layer.

Thread-safe: the pipeline writes from a daemon thread; FastAPI reads from
async handlers.  All mutations go through a single Lock.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Event:
    timestamp: float
    event_type: str     # "chalking" | "sweeper"
    confidence: float
    description: str


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_frame: Optional[bytes] = None   # JPEG bytes, annotated
        self.events: deque[Event] = deque(maxlen=50)
        self._total_chalking: int = 0
        self._total_sweeper: int = 0
        self._last_chalking: Optional[float] = None
        self._last_sweeper: Optional[float] = None
        self.sweep_window_active: bool = False
        self.pipeline_running: bool = False
        self._start_time: float = time.monotonic()

    # ── Pipeline writes ───────────────────────────────────────────────────────

    def push_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self.latest_frame = jpeg

    def record_alert(self, event_type: str, confidence: float, description: str) -> None:
        with self._lock:
            self.events.appendleft(
                Event(
                    timestamp=time.time(),
                    event_type=event_type,
                    confidence=confidence,
                    description=description,
                )
            )
            if event_type == "chalking":
                self._total_chalking += 1
                self._last_chalking = time.time()
            elif event_type == "sweeper":
                self._total_sweeper += 1
                self._last_sweeper = time.time()

    # ── Web reads ─────────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_frame

    def get_stats(self) -> dict:
        with self._lock:
            uptime = int(time.monotonic() - self._start_time)
            return {
                "pipeline_running": self.pipeline_running,
                "sweep_window_active": self.sweep_window_active,
                "total_chalking": self._total_chalking,
                "total_sweeper": self._total_sweeper,
                "last_chalking": self._last_chalking,
                "last_sweeper": self._last_sweeper,
                "uptime_seconds": uptime,
            }

    def get_events(self, limit: int = 30) -> list[dict]:
        with self._lock:
            return [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "confidence": e.confidence,
                    "description": e.description,
                }
                for e in list(self.events)[:limit]
            ]
