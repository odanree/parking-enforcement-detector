"""Parking-enforcement vehicle stopped-in-zone detector.

Signature: a car that *enters* the zone while moving (displacement over the
first `entry_frames` frames > `entry_min_px`) then *stops* (per-frame
displacement < `stop_px_per_frame` for `sustained_frames` consecutive frames).

Cars that were already stationary when first detected are ignored — they are
ordinary parked vehicles, not an officer who just pulled up.
"""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

_ENTRY = "entry"
_WATCHING = "watching"
_SKIP = "skip"          # already-parked car; never trigger


class PEVehicleAnalyzer:
    def __init__(
        self,
        entry_frames: int = 5,
        entry_min_px: float = 15.0,
        stop_px_per_frame: float = 3.0,
        sustained_frames: int = 30,
        cooldown_seconds: float = 120.0,
        position_cooldown_radius: int = 80,
    ) -> None:
        self._entry_frames = entry_frames
        self._entry_min_px = entry_min_px
        self._stop_px = stop_px_per_frame
        self._sustained = sustained_frames
        self._cooldown = cooldown_seconds
        self._pos_radius = position_cooldown_radius

        self._phase: dict[int, str] = {}
        self._entry_centers: dict[int, list[tuple[int, int]]] = {}
        self._prev_center: dict[int, tuple[int, int]] = {}
        self._stop_count: dict[int, int] = {}
        self._last_alert: dict[int, float] = {}
        # Position-based cooldown: [(center, expire_monotonic_time), ...]
        # Survives track-ID resets caused by passing vehicles.
        self._pos_cooldowns: list[tuple[tuple[int, int], float]] = []

    def _on_pos_cooldown(self, center: tuple[int, int]) -> bool:
        now = time.monotonic()
        self._pos_cooldowns = [(c, t) for c, t in self._pos_cooldowns if t > now]
        return any(
            math.hypot(center[0] - c[0], center[1] - c[1]) < self._pos_radius
            for c, _ in self._pos_cooldowns
        )

    def _record_pos_cooldown(self, center: tuple[int, int]) -> None:
        self._pos_cooldowns.append((center, time.monotonic() + self._cooldown))

    def update(self, track_id: int, center: tuple[int, int]) -> bool:
        """Feed one frame's center position. Returns True when the stopped
        signature is confirmed and the cooldown has elapsed."""
        phase = self._phase.get(track_id, _ENTRY)

        if phase == _SKIP:
            return False

        if phase == _ENTRY:
            entry = self._entry_centers.setdefault(track_id, [])
            if not entry and self._on_pos_cooldown(center):
                # New track at a location we just analyzed — skip immediately.
                self._phase[track_id] = _SKIP
                logger.debug("PE vehicle track=%d position on cooldown — skipping", track_id)
                return False
            entry.append(center)
            if len(entry) >= self._entry_frames:
                # Directional displacement: first → last center.
                # Bbox jitter on a stationary car oscillates near zero here;
                # a car actually pulling up shows a clear net movement.
                dx = entry[-1][0] - entry[0][0]
                dy = entry[-1][1] - entry[0][1]
                displacement = (dx * dx + dy * dy) ** 0.5
                if displacement >= self._entry_min_px:
                    self._phase[track_id] = _WATCHING
                    self._prev_center[track_id] = center
                    self._stop_count[track_id] = 0
                    logger.debug(
                        "PE vehicle track=%d entered moving (disp=%.0f px) → watching",
                        track_id, displacement,
                    )
                else:
                    self._phase[track_id] = _SKIP
                    logger.debug(
                        "PE vehicle track=%d was already stationary (disp=%.0f px) — skipping",
                        track_id, displacement,
                    )
            return False

        # phase == _WATCHING
        prev = self._prev_center.get(track_id, center)
        dx = center[0] - prev[0]
        dy = center[1] - prev[1]
        velocity = (dx * dx + dy * dy) ** 0.5
        self._prev_center[track_id] = center

        if velocity < self._stop_px:
            self._stop_count[track_id] = self._stop_count.get(track_id, 0) + 1
        else:
            self._stop_count[track_id] = 0

        if self._stop_count.get(track_id, 0) >= self._sustained:
            now = time.monotonic()
            if now - self._last_alert.get(track_id, 0) >= self._cooldown:
                self._last_alert[track_id] = now
                self._stop_count[track_id] = 0
                self._record_pos_cooldown(center)
                logger.info(
                    "PE vehicle signature — track %d stopped for %d frames",
                    track_id, self._sustained,
                )
                return True

        return False

    def evict(self, track_id: int) -> None:
        for store in (
            self._phase, self._entry_centers, self._prev_center,
            self._stop_count, self._last_alert,
        ):
            store.pop(track_id, None)
