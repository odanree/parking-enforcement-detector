"""Shared in-process state between the detection pipeline and the web layer.

Thread-safe: the pipeline writes from a daemon thread; FastAPI reads from
async handlers.  All mutations go through a single Lock.
"""

from __future__ import annotations

import json
import os
import time
import threading
import uuid
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Any, Deque
from dataclasses import dataclass, field
from typing import Optional

_DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
_SESSION_TIMEOUT: float = float(os.getenv("SESSION_WINDOW_MINUTES", "5")) * 60


@dataclass
class Session:
    session_id: str
    camera_id: int
    started_at: float
    last_seen: float
    event_count: int = 0
    track_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "camera_id":   self.camera_id,
            "started_at":  self.started_at,
            "last_seen":   self.last_seen,
            "event_count": self.event_count,
            "track_ids":   self.track_ids,
            "duration_seconds": round(self.last_seen - self.started_at),
        }


@dataclass
class Event:
    timestamp: float
    event_type: str     # "chalking" | "sweeper" | "pe_vehicle"
    confidence: float
    description: str
    snapshot: Optional[str] = None   # filename only, e.g. "chalking_20240101_120000.jpg"
    frames: list = field(default_factory=list)  # base64 JPEGs for animation
    vote: Optional[str] = None       # "up" | "down" | "archive" | None
    track_id: Optional[int] = None
    session_id: Optional[str] = None
    rag_neighbors: list = field(default_factory=list)   # in-memory only, not persisted
    pipeline_trace: Optional[dict] = None               # in-memory only, not persisted


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
        self._people_alerts: deque[dict[str, Any]] = deque(maxlen=100)
        self._vlm_calls: int = 0
        self._vlm_cost_usd: float = 0.0
        self._sessions: dict[str, Session] = {}  # session_id -> Session
        self._suppressed_sessions: set[str] = set()  # downvoted — skip VLM
        self._trusted_sessions: set[str] = set()     # upvoted — auto-confirm
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
                ev = Event(
                    timestamp=float(d["timestamp"]),
                    event_type=d["event_type"],
                    confidence=float(d["confidence"]),
                    description=d.get("description", ""),
                    snapshot=d.get("snapshot"),
                    frames=[],
                    vote=d.get("vote"),
                    track_id=d.get("track_id"),
                    session_id=d.get("session_id"),
                )
                self.events.appendleft(ev)
                # Rebuild session index from persisted session_id
                sid = d.get("session_id")
                if sid:
                    ts = float(d["timestamp"])
                    tid = d.get("track_id")
                    if sid not in self._sessions:
                        self._sessions[sid] = Session(
                            session_id=sid,
                            camera_id=self.camera_id,
                            started_at=ts,
                            last_seen=ts,
                            event_count=0,
                        )
                    s = self._sessions[sid]
                    s.last_seen = max(s.last_seen, ts)
                    s.event_count += 1
                    if tid is not None and tid not in s.track_ids:
                        s.track_ids.append(tid)
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

    def seek_to_timestamp(self, unix_ts: float) -> str:
        """Jump NVR stream to the given Unix timestamp (RTSP only).
        Returns 'ok', 'no_stream', or 'not_rtsp'.
        """
        if not self._stream:
            return 'no_stream'
        if not hasattr(self._stream, 'seek_to_datetime'):
            return 'not_rtsp'
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc).astimezone()
        env_key = 'AMCREST_CHANNEL' if self.camera_id == 0 else f'AMCREST_CHANNEL_{self.camera_id + 1}'
        nvr_ch_str = os.getenv(env_key)
        nvr_channel = int(nvr_ch_str) if nvr_ch_str else None
        self._stream.seek_to_datetime(dt, nvr_channel=nvr_channel)
        return 'ok'

    def go_live(self) -> str:
        if not self._stream:
            return 'no_stream'
        if not hasattr(self._stream, 'go_live'):
            return 'not_rtsp'
        self._stream.go_live()
        return 'ok'

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
        suppressed_by_session: str | None = None,
        rag_neighbors: list | None = None,
        pipeline_trace: dict | None = None,
        rejection_reason: str | None = None,
        track_id: int | None = None,
    ) -> None:
        with self._lock:
            self._debug_rejected.appendleft({
                "kind": kind,
                "thumbnail": thumbnail_b64,
                "confidence": confidence,
                "description": description,
                "timestamp": time.time(),
                "frames": frames or [],
                "suppressed_by_session": suppressed_by_session,
                "rag_neighbors": rag_neighbors or [],
                "pipeline_trace": pipeline_trace,
                "rejection_reason": rejection_reason,
                "track_id": track_id,
            })

    def get_rejected_vlm(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._debug_rejected)

    def clear_rejected_vlm(self) -> None:
        with self._lock:
            self._debug_rejected.clear()

    def record_people_alert(
        self,
        confidence: float,
        description: str,
        frames: list[str] | None = None,
        timestamp: float | None = None,
        track_id: int | None = None,
        rag_neighbors: list | None = None,
        pipeline_trace: dict | None = None,
        thumbnail: str | None = None,
    ) -> None:
        with self._lock:
            self._people_alerts.appendleft({
                "confidence":    confidence,
                "description":   description,
                "frames":        frames or [],
                "timestamp":     timestamp if timestamp is not None else time.time(),
                "track_id":      track_id,
                "rag_neighbors": rag_neighbors or [],
                "pipeline_trace": pipeline_trace,
                "thumbnail":     thumbnail,
            })

    def get_people_alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._people_alerts)[:limit]

    def _get_or_create_session(self, ts: float, track_id: int | None) -> str:
        """Return session_id for this event, joining an open session or starting a new one.
        Must be called with self._lock held.
        """
        open_sessions = [
            s for s in self._sessions.values()
            if s.camera_id == self.camera_id and ts - s.last_seen < _SESSION_TIMEOUT
        ]
        if open_sessions:
            session = max(open_sessions, key=lambda s: s.last_seen)
            session.last_seen = ts
            session.event_count += 1
            if track_id is not None and track_id not in session.track_ids:
                session.track_ids.append(track_id)
            return session.session_id
        sid = uuid.uuid4().hex[:6]
        self._sessions[sid] = Session(
            session_id=sid,
            camera_id=self.camera_id,
            started_at=ts,
            last_seen=ts,
            event_count=1,
            track_ids=[track_id] if track_id is not None else [],
        )
        return sid

    def get_sessions(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in sorted(self._sessions.values(), key=lambda s: s.started_at, reverse=True)]

    def find_open_suppressed_session(self) -> str | None:
        """Return session_id if a downvoted session is still within its window."""
        ts = time.time()
        with self._lock:
            for sid in self._suppressed_sessions:
                s = self._sessions.get(sid)
                if s and s.camera_id == self.camera_id and ts - s.last_seen < _SESSION_TIMEOUT:
                    return sid
        return None

    def find_open_trusted_session(self) -> str | None:
        """Return session_id if an upvoted session is still within its window."""
        ts = time.time()
        with self._lock:
            for sid in self._trusted_sessions:
                s = self._sessions.get(sid)
                if s and s.camera_id == self.camera_id and ts - s.last_seen < _SESSION_TIMEOUT:
                    return sid
        return None

    def record_vlm_usage(self, usage: dict | None) -> None:
        cost = _estimate_cost(usage) if usage else 0.0
        with self._lock:
            self._vlm_calls += 1
            self._vlm_cost_usd += cost

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
        track_id: int | None = None,
        timestamp: float | None = None,
        rag_neighbors: list | None = None,
        pipeline_trace: dict | None = None,
    ) -> None:
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            session_id = self._get_or_create_session(ts, track_id) if event_type == "chalking" else None
            self.events.appendleft(
                Event(
                    timestamp=ts,
                    event_type=event_type,
                    confidence=confidence,
                    description=description,
                    snapshot=snapshot,
                    frames=frames or [],
                    track_id=track_id,
                    session_id=session_id,
                    rag_neighbors=rag_neighbors or [],
                    pipeline_trace=pipeline_trace,
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
                "track_id":   track_id,
                "session_id": session_id,
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
                "is_live": not (self._stream and (hasattr(self._stream, '_speed') or getattr(self._stream, 'is_playback', False))),
                "is_nvr_playback": getattr(self._stream, 'is_playback', False),
                "fps": float(len(self._fps_times)),
                "demo_mode": _DEMO_MODE,
                "vlm_calls": self._vlm_calls,
                "vlm_cost_usd": round(self._vlm_cost_usd, 5),
            }

    def vote_event(self, timestamp: float, vote: str | None) -> bool:
        """Set the vote on the event matching timestamp. Returns True if found.

        vote='archive' removes the event from both the deque and the JSONL file.
        Downvoting a session marks it suppressed (future VLM calls skipped).
        Upvoting a session marks it trusted (future detections auto-confirmed).
        """
        _VALID_VOTES = {"up", "down", "archive", None}
        if vote not in _VALID_VOTES:
            raise ValueError(f"vote must be one of {_VALID_VOTES}")
        with self._lock:
            match = next((ev for ev in self.events if abs(ev.timestamp - timestamp) < 0.001), None)
            if match is None:
                return False
            if vote == "archive":
                keep = deque((e for e in self.events if e is not match), maxlen=self.events.maxlen)
                self.events = keep
                if match.event_type == "chalking":
                    self._total_chalking = max(0, self._total_chalking - 1)
                elif match.event_type == "sweeper":
                    self._total_sweeper = max(0, self._total_sweeper - 1)
                if match.session_id:
                    self._suppressed_sessions.discard(match.session_id)
                    self._trusted_sessions.discard(match.session_id)
            else:
                match.vote = vote
                if match.session_id:
                    if vote == "down":
                        self._suppressed_sessions.add(match.session_id)
                        self._trusted_sessions.discard(match.session_id)
                    elif vote == "up":
                        self._trusted_sessions.add(match.session_id)
                        self._suppressed_sessions.discard(match.session_id)
                    else:  # None — vote toggled off
                        self._suppressed_sessions.discard(match.session_id)
                        self._trusted_sessions.discard(match.session_id)
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
                    "track_id":   ev.track_id,
                    "session_id": ev.session_id,
                }
                if ev.vote is not None:
                    entry["vote"] = ev.vote
                lines.append(json.dumps(entry))
            _LOGS_DIR.mkdir(exist_ok=True)
            self._events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def get_vlm_usage(self) -> dict:
        with self._lock:
            return {"calls": self._vlm_calls, "cost_usd": round(self._vlm_cost_usd, 5)}

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
                    "track_id": e.track_id,
                    "session_id": e.session_id,
                    "camera_id": self.camera_id,
                    "rag_neighbors": e.rag_neighbors,
                    "pipeline_trace": e.pipeline_trace,
                }
                for e in list(self.events)[:limit]
            ]

    def clear_events(self) -> None:
        with self._lock:
            self.events.clear()
            self._people_alerts.clear()
            self._total_chalking = 0
            self._total_sweeper  = 0
            self._last_chalking  = None
            self._last_sweeper   = None


# Anthropic pricing per million tokens (as of 2025)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_creation": 0.80},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_creation": 18.75},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75}


def _estimate_cost(usage: dict) -> float:
    p = _MODEL_PRICING.get(usage.get("model", ""), _DEFAULT_PRICING)
    mtok = 1_000_000
    return (
        usage.get("input_tokens", 0)              * p["input"]           / mtok
        + usage.get("output_tokens", 0)           * p["output"]          / mtok
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]      / mtok
        + usage.get("cache_creation_input_tokens", 0) * p["cache_creation"] / mtok
    )
