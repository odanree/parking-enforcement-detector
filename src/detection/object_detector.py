from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class names the spec requires us to track
_TRACKED_CLASSES = {"person", "truck", "motorcycle", "car"}

# Cars are exempt from stationary masking — a car that *stops* is the signal
# we want for PE vehicle detection, not noise to suppress.
_STATIONARY_EXEMPT = {"car"}

# Classes that get position-grid suppression: if a grid cell has been
# continuously occupied by one of these classes for _GRID_SUPPRESS_FRAMES frames,
# that cell is permanently blacklisted (survives track-ID churn from the tracker).
_POSITION_SUPPRESS_CLASSES = {"person"}


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int] = field(init=False)
    area_fraction: float = 0.0

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]


class ObjectDetector:
    """YOLOv8 tracker wrapper with built-in stationary-mask suppression.

    Stationary masking: if a tracked object hasn't moved more than
    `stationary_px` pixels in `stationary_frames` consecutive frames it is
    excluded from the returned detections so a parked car never re-fires.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        threshold: float = 0.65,
        input_size: tuple[int, int] = (1280, 720),
        min_area_fraction: float = 0.02,
        max_area_fraction: float = 0.50,
        stationary_px: int = 15,
        stationary_frames: int = 30,
        tracker_config: str = "config/bytetrack.yaml",
        fp_grid_cell_px: int = 48,
        fp_suppress_seconds: float = 10.0,
    ) -> None:
        self._model = YOLO(model_path)
        self._threshold = threshold
        # ByteTrack low threshold: fed to the tracker so it can re-associate
        # partly-occluded tracks without creating new false-positive tracks.
        self._track_low_thresh = max(0.05, threshold * 0.40)
        self._tracker_config = tracker_config
        self._input_size = input_size
        self._min_area = min_area_fraction
        self._max_area = max_area_fraction
        self._stationary_px = stationary_px
        self._stationary_frames = stationary_frames

        # track_id -> deque of recent centers
        self._position_history: dict[int, list[tuple[int, int]]] = {}

        frame_px = input_size[0] * input_size[1]
        self._min_px = min_area_fraction * frame_px
        self._max_px = max_area_fraction * frame_px

        # Position-grid false-positive suppressor.
        # Divides the frame into cells of fp_grid_cell_px × fp_grid_cell_px.
        # If a _POSITION_SUPPRESS_CLASSES detection occupies the same cell
        # continuously for fp_suppress_seconds (time-based, FPS-independent),
        # the cell is permanently blacklisted — fire hydrants / trash cans silenced.
        self._grid_cell_px = fp_grid_cell_px
        self._fp_suppress_seconds = fp_suppress_seconds
        self._grid_first_seen: dict[tuple[int, int], float] = {}  # cell -> monotonic start time
        self._grid_blocked: set[tuple[int, int]] = set()          # permanently suppressed cells

        logger.info(
            "Detector ready — model=%s high=%.2f low=%.2f tracker=%s fp_grid=%dpx suppress_after=%.1fs",
            model_path, threshold, self._track_low_thresh, tracker_config,
            fp_grid_cell_px, fp_suppress_seconds,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self._input_size) if (w, h) != self._input_size else frame

        results = self._model.track(
            resized,
            persist=True,
            conf=self._track_low_thresh,
            tracker=self._tracker_config,
            classes=self._class_ids(),
            verbose=False,
        )

        detections: list[Detection] = []
        frame_area = self._input_size[0] * self._input_size[1]

        if results[0].boxes.id is None:
            self._update_grid(set())   # no hits → reset all active streaks
            return detections

        hits_this_frame: set[tuple[int, int]] = set()

        for box, track_id, conf, cls in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.id.cpu().numpy().astype(int),
            results[0].boxes.conf.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy().astype(int),
        ):
            class_name = self._model.names[cls]
            if class_name not in _TRACKED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            area_frac = area / frame_area

            if not (self._min_px <= area <= self._max_px):
                continue

            # Post-track confidence gate: ByteTrack runs at _track_low_thresh so
            # it can re-associate partly-occluded objects, but downstream stages
            # should only see detections that clear the user-configured threshold.
            # Without this, ~70 % of stage-1 VLM calls were spent rejecting sub-0.30
            # YOLO hits on shadows.
            if float(conf) < self._threshold:
                continue

            det = Detection(
                track_id=int(track_id),
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                area_fraction=area_frac,
            )

            if class_name not in _STATIONARY_EXEMPT and self._is_stationary(det.track_id, det.center):
                continue

            # Position-grid FP suppression: skip permanently blocked cells.
            if class_name in _POSITION_SUPPRESS_CLASSES:
                cell = self._grid_cell(det.center)
                if cell in self._grid_blocked:
                    continue
                hits_this_frame.add(cell)

            detections.append(det)

        self._update_grid(hits_this_frame)
        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _class_ids(self) -> list[int]:
        return [
            idx
            for idx, name in self._model.names.items()
            if name in _TRACKED_CLASSES
        ]

    def _grid_cell(self, center: tuple[int, int]) -> tuple[int, int]:
        return (center[0] // self._grid_cell_px, center[1] // self._grid_cell_px)

    def _update_grid(self, hits: set[tuple[int, int]]) -> None:
        now = time.monotonic()
        # Advance clocks for active cells; block those that exceed the threshold.
        for cell in hits:
            if cell not in self._grid_first_seen:
                self._grid_first_seen[cell] = now
            elif now - self._grid_first_seen[cell] >= self._fp_suppress_seconds:
                if cell not in self._grid_blocked:
                    logger.info(
                        "FP grid cell (%d,%d) blocked after %.1fs — likely static object",
                        cell[0], cell[1], now - self._grid_first_seen[cell],
                    )
                self._grid_blocked.add(cell)
        # Reset clock for cells that had no hit this frame (object moved away).
        for cell in list(self._grid_first_seen):
            if cell not in hits and cell not in self._grid_blocked:
                del self._grid_first_seen[cell]

    def clear_fp_grid(self) -> None:
        """Reset all learned FP suppressions (e.g. after camera repositioning)."""
        self._grid_first_seen.clear()
        self._grid_blocked.clear()
        logger.info("FP position grid cleared")

    def _is_stationary(self, track_id: int, center: tuple[int, int]) -> bool:
        history = self._position_history.setdefault(track_id, [])
        history.append(center)
        if len(history) > self._stationary_frames:
            history.pop(0)

        if len(history) < self._stationary_frames:
            return False

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return spread <= self._stationary_px
