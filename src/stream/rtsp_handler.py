import os
import queue
import threading
import time
import logging
import cv2

logger = logging.getLogger(__name__)

# Force TCP transport before any VideoCapture is opened.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class RTSPHandler:
    """Thread-safe RTSP frame producer.

    Drops stale frames when the consumer falls behind so the pipeline always
    sees the most recent image rather than a growing backlog.
    """

    def __init__(self, url: str, queue_size: int = 2, reconnect_delay: float = 3.0):
        self._url = url
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._reconnect_delay = reconnect_delay
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtsp-reader")
        self._thread.start()
        logger.info("RTSP reader started → %s", self._url)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)

    def get_frame(self, timeout: float = 2.0):
        """Return the latest frame or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        cap = self._open_capture()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Stream read failed — reconnecting in %.0fs…", self._reconnect_delay)
                cap.release()
                time.sleep(self._reconnect_delay)
                cap = self._open_capture()
                continue

            # Evict the stale frame so we never block
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

        cap.release()
