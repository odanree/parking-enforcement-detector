"""Real-time action classifier for tracked persons.

Classifies each tracked person into one of: standing, walking, running,
crouching, bending — using only bbox velocity (from ByteTrack history) and
optional PoseSignals. No ML inference; runs every frame with negligible cost.

Velocity is normalised by person height so the thresholds are scale-invariant
across near/far subjects.  Label smoothing via a rolling majority vote prevents
flicker when a person briefly pauses mid-stride.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.detection.pose_estimator import PoseSignals


# ── Tunable thresholds ────────────────────────────────────────────────────────
# Velocity expressed as (bbox-centre displacement per frame) / (person height).
# At ~15 fps a walking adult moves ~2–5 % of their height per frame; running
# is ~8–15 %.  Standing jitter is typically <1 %.
_WALK_THRESHOLD  = 0.015   # body-heights / frame — above = walking
_RUN_THRESHOLD   = 0.075   # body-heights / frame — above = running

_SMOOTH_WINDOW   = 10      # frames of history for majority-vote smoothing
_EVICT_FRAMES    = 60      # frames of absence before dropping a track


@dataclass
class _TrackState:
    # (cx, cy, height) per frame — used to compute velocity
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    # raw per-frame label before smoothing
    raw: deque = field(default_factory=lambda: deque(maxlen=_SMOOTH_WINDOW))
    label: str = "standing"
    miss: int = 0


class ActionClassifier:
    """Per-track action label with temporal smoothing.

    Call ``update()`` once per detection per frame; call ``tick_absent()`` at
    the end of each frame with the set of currently-visible track ids.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        pose: "PoseSignals | None" = None,
    ) -> str:
        """Classify and return the smoothed action label for this track."""
        st = self._tracks.setdefault(track_id, _TrackState())
        st.miss = 0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h  = max(1, y2 - y1)
        st.history.append((cx, cy, h))

        # ── Pose-based labels (take priority over motion) ─────────────────
        if pose is not None and pose.confidence > 0.25:
            if pose.is_crouching:
                raw = "crouching"
            elif pose.hand_low:
                raw = "bending"
            else:
                raw = self._motion_label(st)
        else:
            raw = self._motion_label(st)

        st.raw.append(raw)
        st.label = self._majority(st.raw)
        return st.label

    def label(self, track_id: int) -> str:
        st = self._tracks.get(track_id)
        return st.label if st else "standing"

    def tick_absent(self, active_ids: set[int]) -> None:
        """Evict tracks that have not been seen for a while."""
        for tid in list(self._tracks):
            if tid not in active_ids:
                self._tracks[tid].miss += 1
                if self._tracks[tid].miss >= _EVICT_FRAMES:
                    del self._tracks[tid]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _motion_label(self, st: _TrackState) -> str:
        if len(st.history) < 3:
            return "standing"
        pts = list(st.history)
        # velocity = mean frame-to-frame displacement over recent history
        disps = [
            math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]) / pts[i][2]
            for i in range(1, len(pts))
        ]
        vel = sum(disps) / len(disps)
        if vel >= _RUN_THRESHOLD:
            return "running"
        if vel >= _WALK_THRESHOLD:
            return "walking"
        return "standing"

    @staticmethod
    def _majority(raw: deque) -> str:
        counts: dict[str, int] = {}
        for v in raw:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.__getitem__)
