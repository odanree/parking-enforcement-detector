"""Video file source — drop-in replacement for RTSPHandler during testing.

Reads frames from a local video file at the camera's native fps (or a
configurable playback speed multiplier).  Loops automatically so the clip
keeps cycling without restarting the pipeline.

Usage: set VIDEO_PATH=/path/to/clip.mp4 in .env — pipeline.py picks it up.
"""

from __future__ import annotations

import queue
import threading
import time
import logging

import cv2

logger = logging.getLogger(__name__)


class VideoFileHandler:
    def __init__(
        self,
        path: str,
        loop: bool = True,
        speed: float = 1.0,
        queue_size: int = 2,
    ) -> None:
        self._path = path
        self._loop = loop
        self._speed = speed
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop_fn, daemon=True, name="video-file")
        self._thread.start()
        mode = "looping" if self._loop else "single-pass"
        logger.info("Video file source (%s) → %s  speed=%.1fx", mode, self._path, self._speed)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_frame(self, timeout: float = 2.0):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _loop_fn(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._path)
            if not cap.isOpened():
                logger.error("Cannot open video file: %s", self._path)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / (fps * self._speed)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info("Loaded %s  %.0f fps  %d frames", self._path, fps, total)

            while not self._stop.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()

                if not ok:
                    break   # end of file — outer loop will restart if looping

                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put(frame)

                elapsed = time.monotonic() - t0
                sleep = frame_delay - elapsed
                if sleep > 0:
                    time.sleep(sleep)

            cap.release()

            if not self._loop:
                logger.info("Video file finished — pipeline will idle")
                break

            logger.info("Video file looping…")
