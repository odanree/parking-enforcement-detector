from __future__ import annotations

import time
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class SweeperAnalyzer:
    """Detects the street-sweeper behavioral signature.

    Two gate conditions must both be true before VLM analysis is triggered:

    1. **Schedule gate** — current day/time falls within a configured sweep
       window (e.g. Tuesday 08:00–13:00).  Keeps the pipeline idle the rest
       of the week so no CPU/API is wasted.

    2. **Velocity gate** — a tracked truck/vehicle's pixel-per-frame speed
       stays within [min_velocity, max_velocity] for `sustained_frames`
       consecutive frames, matching the 5–10 mph sweeper cadence.
    """

    def __init__(
        self,
        schedule: list[dict],          # from schedule.yaml
        min_velocity: float = 3.0,     # px/frame
        max_velocity: float = 20.0,    # px/frame
        sustained_frames: int = 10,
    ) -> None:
        self._schedule = schedule
        self._min_v = min_velocity
        self._max_v = max_velocity
        self._sustained = sustained_frames

        # track_id -> deque of recent centers
        self._centers: dict[int, deque[tuple[int, int]]] = {}
        # track_id -> consecutive frames within velocity range
        self._in_range_count: dict[int, int] = {}
        # track_id -> timestamp of last alert
        self._last_alert: dict[int, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def is_sweep_window(self, now: datetime | None = None) -> bool:
        """Return True if the current moment is within a configured sweep window."""
        now = now or datetime.now()
        for window in self._schedule:
            if now.weekday() != window["weekday"]:
                continue
            start = datetime.strptime(window["start_time"], "%H:%M").time()
            end = datetime.strptime(window["end_time"], "%H:%M").time()
            if start <= now.time() <= end:
                return True
        return False

    def update(self, track_id: int, center: tuple[int, int]) -> bool:
        """Feed one frame's object center.  Returns True when the sweeper
        velocity signature is confirmed and a cooldown has elapsed."""
        history = self._centers.setdefault(track_id, deque(maxlen=2))
        history.append(center)

        if len(history) < 2:
            return False

        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        velocity = (dx ** 2 + dy ** 2) ** 0.5   # px/frame

        if self._min_v <= velocity <= self._max_v:
            self._in_range_count[track_id] = self._in_range_count.get(track_id, 0) + 1
        else:
            self._in_range_count[track_id] = 0
            return False

        if self._in_range_count[track_id] < self._sustained:
            return False

        now_ts = time.monotonic()
        if now_ts - self._last_alert.get(track_id, 0) < 600:   # 10-min cooldown
            return False

        self._last_alert[track_id] = now_ts
        logger.info(
            "Sweeper velocity signature — track %d  %.1f px/frame sustained %d frames",
            track_id,
            velocity,
            self._in_range_count[track_id],
        )
        return True

    def evict(self, track_id: int) -> None:
        self._centers.pop(track_id, None)
        self._in_range_count.pop(track_id, None)
        self._last_alert.pop(track_id, None)
