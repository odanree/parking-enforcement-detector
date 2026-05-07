"""Shared in-process state between the detection pipeline and the web layer.

Thread-safe: the pipeline writes from a daemon thread; FastAPI reads from
async handlers.  All mutations go through a single Lock.
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque
from dataclasses import dataclass, field
from typing import Optional

_DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"


@dataclass
class Event:
    timestamp: float
    event_type: str     # "chalking" | "sweeper" | "pe_vehicle"
    confidence: float
    description: str
    snapshot: Optional[str] = None   # filename only, e.g. "chalking_20240101_120000.jpg"
    frames: list = field(default_factory=list)  # base64 JPEGs for animation
    vote: Optional[str] = None       # "up" | "down" | "archive" | None


_LOGS_DIR = Path("logs")


class AppState:
    def __init__(self, camera_id: int = 0) -> None:
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self.latest_frame: Optional[bytes] = None   # JPEG bytes, annotated
        self.events: deque[Event] = deque(maxlen=50)
        self._events_file = _LOGS_DIR / f"events_{camera_id}.jsonl"
        self._total_chalking: int = 0
        self._total_sweeper: int = 0
        self._last_chalking: Optional[float] = None
        self._last_sweeper: Optional[float] = None
        self.sweep_window_active: bool = False
        self.pipeline_running: bool = False
        self._start_time: float = time.monotonic()
        # Zone polygon in frame coordinates — pipeline watches _zone_version
        self.zone_polygon: list[list[int]] = []
        self._zone_version: int = 0
        self.paused: bool = False
        self.motion_detect_enabled: bool = False
        self.privacy_mode: bool = _DEMO_MODE
        self.privacy_regions: list[list[int]] = []  # [[x1,y1,x2,y2], ...]
        self._stream = None   # VideoFileHandler reference, set by pipeline
        self._fps_times: Deque[float] = deque()  # monotonic timestamps of recent frames
        self._pending_vlm: list[dict[str, Any]] = []  # in-flight VLM jobs
        self._vlm_sample_counts: dict[str, int] = {}  # job_id → cumulative sample count
        self._debug_rejected: deque[dict[str, Any]] = deque(maxlen=30)
        self._load_events()

    def _load_events(self) -> None:
        if not self._events_file.exists():
            return
        try:
            lines = [l for l in self._events_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            return
        # Rebuild total counts from full history
        for line in lines:
            try:
                d = json.loads(line)
                et = d.get("event_type", "")
                ts = float(d.get("timestamp", 0))
                if et == "chalking":
                    self._total_chalking += 1
                    if self._last_chalking is None or ts > self._last_chalking:
                        self._last_chalking = ts
                elif et == "sweeper":
                    self._total_sweeper += 1
                    if self._last_sweeper is None or ts > self._last_sweeper:
                        self._last_sweeper = ts
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        # Load the most recent 50 into the dashboard deque (frames not persisted)
        for line in lines[-50:]:
            try:
                d = json.loads(line)
                self.events.appendleft(Event(
                    timestamp=float(d["timestamp"]),
                    event_type=d["event_type"],
                    confidence=float(d["confidence"]),
                    description=d.get("description", ""),
                    snapshot=d.get("snapshot"),
                    frames=[],
                    vote=d.get("vote"),
                ))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    # ── Playback control ─────────────────────────────────────────────────────

    def set_stream(self, stream) -> None:
        self._stream = stream

    def set_playback_speed(self, speed: float) -> float:
        speed = max(0.1, min(16.0, speed))
        if self._stream and hasattr(self._stream, 'set_speed'):
            self._stream.set_speed(speed)
        return speed

    def get_playback_speed(self) -> float:
        if self._stream and hasattr(self._stream, '_speed'):
            return self._stream._speed
        return 1.0

    def seek_playback(self, seconds: float) -> None:
        """Seek by ±seconds relative to current position."""
        if self._stream and hasattr(self._stream, 'seek'):
            fps = getattr(self._stream, '_current_fps', 20.0)
            self._stream.seek(int(seconds * fps))

    def set_playback_direction(self, direction: int) -> int:
        direction = 1 if direction >= 0 else -1
        if self._stream and hasattr(self._stream, 'set_direction'):
            self._stream.set_direction(direction)
        return direction

    def get_playback_direction(self) -> int:
        if self._stream and hasattr(self._stream, '_direction'):
            return self._stream._direction
        return 1

    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            if self._stream and hasattr(self._stream, 'pause'):
                if self.paused:
                    self._stream.pause()
                else:
                    self._stream.resume()
            return self.paused

    def toggle_motion_detect(self) -> bool:
        with self._lock:
            self.motion_detect_enabled = not self.motion_detect_enabled
            return self.motion_detect_enabled

    def toggle_privacy(self) -> bool:
        with self._lock:
            self.privacy_mode = not self.privacy_mode
            return self.privacy_mode

    def update_privacy_regions(self, regions: list[list[int]]) -> None:
        with self._lock:
            self.privacy_regions = regions

    def get_privacy_regions(self) -> list[list[int]]:
        with self._lock:
            return list(self.privacy_regions)

    # ── VLM pending queue ─────────────────────────────────────────────────────

    def add_pending_vlm(self, kind: str, track_id: int, thumbnail_b64: str) -> None:
        job_id = f"{kind}_{track_id}"
        with self._lock:
            self._vlm_sample_counts[job_id] = self._vlm_sample_counts.get(job_id, 0) + 1
            self._pending_vlm = [j for j in self._pending_vlm if j["id"] != job_id]
            self._pending_vlm.append({
                "id": job_id,
                "kind": kind,
                "thumbnail": thumbnail_b64,
                "submitted_at": time.time(),
                "sample_num": self._vlm_sample_counts[job_id],
            })

    def record_rejected_vlm(
        self, kind: str, thumbnail_b64: str, confidence: float, description: str,
        frames: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._debug_rejected.appendleft({
                "kind": kind,
                "thumbnail": thumbnail_b64,
                "confidence": confidence,
                "description": description,
                "timestamp": time.time(),
                "frames": frames or [],
            })

    def get_rejected_vlm(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._debug_rejected)

    def clear_rejected_vlm(self) -> None:
        with self._lock:
            self._debug_rejected.clear()

    def complete_pending_vlm(self, kind: str, track_id: int, detected: bool) -> None:
        job_id = f"{kind}_{track_id}"
        with self._lock:
            for j in self._pending_vlm:
                if j["id"] == job_id:
                    j["completed_at"] = time.time()
                    j["detected"] = detected
                    break

    def get_pending_vlm(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            # Drop entries that completed more than 4 seconds ago
            self._pending_vlm = [
                j for j in self._pending_vlm
                if "completed_at" not in j or now - j["completed_at"] < 4.0
            ]
            return list(self._pending_vlm)

    # ── Zone ─────────────────────────────────────────────────────────────────

    def update_zone(self, polygon: list[list[int]]) -> None:
        with self._lock:
            self.zone_polygon = polygon
            self._zone_version += 1

    def get_zone_version(self) -> int:
        with self._lock:
            return self._zone_version

    # ── Pipeline writes ───────────────────────────────────────────────────────

    def tick_fps(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._fps_times.append(now)
            cutoff = now - 1.0  # rolling 1-second window
            while self._fps_times and self._fps_times[0] < cutoff:
                self._fps_times.popleft()

    def get_fps(self) -> float:
        with self._lock:
            return float(len(self._fps_times))

    def push_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self.latest_frame = jpeg

    def record_alert(
        self,
        event_type: str,
        confidence: float,
        description: str,
        snapshot: str | None = None,
        frames: list[str] | None = None,
    ) -> None:
        ts = time.time()
        with self._lock:
            self.events.appendleft(
                Event(
                    timestamp=ts,
                    event_type=event_type,
                    confidence=confidence,
                    description=description,
                    snapshot=snapshot,
                    frames=frames or [],
                )
            )
            if event_type == "chalking":
                self._total_chalking += 1
                self._last_chalking = ts
            elif event_type == "sweeper":
                self._total_sweeper += 1
                self._last_sweeper = ts
        # Persist outside the lock — file I/O should not block the pipeline thread
        try:
            _LOGS_DIR.mkdir(exist_ok=True)
            entry = {
                "timestamp":  ts,
                "event_type": event_type,
                "confidence": confidence,
                "description": description,
                "snapshot":   snapshot,
            }
            with self._events_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # non-critical — dashboard still works without persistence

    # ── Web reads ─────────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_frame

    def get_stats(self) -> dict:
        with self._lock:
            uptime = int(time.monotonic() - self._start_time)
            return {
                "pipeline_running": self.pipeline_running,
                "paused": self.paused,
                "motion_detect_enabled": self.motion_detect_enabled,
                "privacy_mode": self.privacy_mode,
                "sweep_window_active": self.sweep_window_active,
                "total_chalking": self._total_chalking,
                "total_sweeper": self._total_sweeper,
                "last_chalking": self._last_chalking,
                "last_sweeper": self._last_sweeper,
                "uptime_seconds": uptime,
                "playback_speed": self._stream._speed if self._stream and hasattr(self._stream, '_speed') else 1.0,
                "playback_direction": self._stream._direction if self._stream and hasattr(self._stream, '_direction') else 1,
                "is_live": not (self._stream and hasattr(self._stream, '_speed')),
                "fps": float(len(self._fps_times)),
                "demo_mode": _DEMO_MODE,
            }

    def vote_event(self, timestamp: float, vote: str | None) -> bool:
        """Set the vote on the event matching timestamp. Returns True if found."""
        _VALID_VOTES = {"up", "down", "archive", None}
        if vote not in _VALID_VOTES:
            raise ValueError(f"vote must be one of {_VALID_VOTES}")
        with self._lock:
            for ev in self.events:
                if abs(ev.timestamp - timestamp) < 0.001:
                    ev.vote = vote
                    break
            else:
                return False
        # Rewrite the JSONL file with the updated vote
        self._rewrite_events_file()
        return True

    def _rewrite_events_file(self) -> None:
        try:
            with self._lock:
                snapshot = list(self.events)
            lines = []
            for ev in reversed(snapshot):  # file is oldest-first
                entry: dict = {
                    "timestamp":  ev.timestamp,
                    "event_type": ev.event_type,
                    "confidence": ev.confidence,
                    "description": ev.description,
                    "snapshot":   ev.snapshot,
                }
                if ev.vote is not None:
                    entry["vote"] = ev.vote
                lines.append(json.dumps(entry))
            _LOGS_DIR.mkdir(exist_ok=True)
            self._events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def get_events(self, limit: int = 30) -> list[dict]:
        with self._lock:
            return [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "confidence": e.confidence,
                    "description": e.description,
                    "snapshot_url": f"/snapshots/{e.snapshot}" if e.snapshot else None,
                    "frames": e.frames,
                    "vote": e.vote,
                }
                for e in list(self.events)[:limit]
            ]
