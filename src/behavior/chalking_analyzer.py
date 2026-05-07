from __future__ import annotations

import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ChalkingAnalyzer:
    """Triggers VLM sampling while a person is present in the zone.

    Replaces the bounding-box height-decrease heuristic, which is unreliable
    at camera distances where the person occupies only a small fraction of the
    frame.  Instead, samples the VLM periodically and lets it detect whether
    the person is holding a chalk stick near a tire.

    A rolling frame buffer keeps the last `frame_buffer_size` wide-crop JPEGs
    per track (sampled every `buffer_sample_every_n` frames).  When a VLM call
    is triggered, the full sequence is sent so the model can use earlier frames
    as context (e.g. person was exiting a vehicle door → not chalking).
    """

    def __init__(
        self,
        entry_frames: int = 10,
        sample_every_n: int = 30,
        cooldown_seconds: float = 60.0,
        frame_buffer_size: int = 5,
        buffer_sample_every_n: int = 3,
    ) -> None:
        self._entry_frames = entry_frames
        self._sample_every_n = sample_every_n
        self._cooldown = cooldown_seconds
        self._frame_buffer_size = frame_buffer_size
        self._buffer_sample_every_n = buffer_sample_every_n

        self._frame_count: dict[int, int] = {}
        self._last_alert: dict[int, float] = {}
        self._frame_buffers: dict[int, deque[bytes]] = {}

    def update(self, track_id: int) -> bool:
        """Returns True when this frame should be sent to the VLM.

        Fires once the person has been stably tracked for `entry_frames`, then
        every `sample_every_n` frames thereafter (unless in cooldown).
        """
        count = self._frame_count.get(track_id, 0) + 1
        self._frame_count[track_id] = count

        if count < self._entry_frames:
            return False

        if (count - self._entry_frames) % self._sample_every_n != 0:
            return False

        if time.monotonic() - self._last_alert.get(track_id, 0) < self._cooldown:
            return False

        return True

    def store_frame(self, track_id: int, jpeg_bytes: bytes) -> None:
        """Add a wide-crop JPEG to the rolling buffer for this track.

        Called every frame; internally samples at `buffer_sample_every_n` so
        the buffer spans a longer time window without storing every frame.
        """
        count = self._frame_count.get(track_id, 0)
        if count % self._buffer_sample_every_n != 0:
            return
        buf = self._frame_buffers.setdefault(
            track_id, deque(maxlen=self._frame_buffer_size)
        )
        buf.append(jpeg_bytes)

    def get_frame_buffer(self, track_id: int) -> list[bytes]:
        """Return buffered frames oldest-first for the given track."""
        return list(self._frame_buffers.get(track_id, []))

    def on_alert(self, track_id: int) -> None:
        """Call after a confirmed chalking alert to start the per-track cooldown."""
        self._last_alert[track_id] = time.monotonic()
        logger.info("Chalking alert confirmed — track %d cooldown started", track_id)

    def evict(self, track_id: int) -> None:
        self._frame_count.pop(track_id, None)
        self._last_alert.pop(track_id, None)
        self._frame_buffers.pop(track_id, None)
