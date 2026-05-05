from __future__ import annotations

import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ChalkingAnalyzer:
    """Detects the chalking behavioral signature.

    Signature: a person in the street zone whose bounding-box height
    decreases by ≥ 30 % relative to their own recent baseline — indicating
    a crouch or forward lean toward a tire.

    Each track_id can only fire once per `cooldown_seconds` to avoid
    duplicate alerts while the officer is still at the same car.
    """

    def __init__(
        self,
        height_decrease_threshold: float = 0.30,
        history_frames: int = 15,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._threshold = height_decrease_threshold
        self._history_frames = history_frames
        self._cooldown = cooldown_seconds

        # track_id -> deque of recent bounding-box heights
        self._heights: dict[int, deque[int]] = {}
        # track_id -> timestamp of last alert
        self._last_alert: dict[int, float] = {}

    def update(self, track_id: int, bbox_height: int) -> bool:
        """Feed one frame's bounding-box height.  Returns True when the
        chalking signature is confirmed for this track."""
        history = self._heights.setdefault(track_id, deque(maxlen=self._history_frames))
        history.append(bbox_height)

        if len(history) < max(4, self._history_frames // 3):
            return False

        # Baseline = average of the first third of the history window
        baseline_window = list(history)[: len(history) // 3]
        baseline = sum(baseline_window) / len(baseline_window)

        if baseline == 0:
            return False

        current = history[-1]
        decrease = (baseline - current) / baseline

        if decrease < self._threshold:
            return False

        now = time.monotonic()
        if now - self._last_alert.get(track_id, 0) < self._cooldown:
            return False

        self._last_alert[track_id] = now
        logger.info(
            "Chalking signature — track %d height ↓%.0f%% (baseline=%d → current=%d)",
            track_id,
            decrease * 100,
            int(baseline),
            current,
        )
        return True

    def evict(self, track_id: int) -> None:
        """Call when a track disappears to free memory."""
        self._heights.pop(track_id, None)
        self._last_alert.pop(track_id, None)
