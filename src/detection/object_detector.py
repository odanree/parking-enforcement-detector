from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class names the spec requires us to track
_TRACKED_CLASSES = {"person", "truck", "motorcycle"}


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
    ) -> None:
        self._model = YOLO(model_path)
        self._threshold = threshold
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

        logger.info("Detector ready — model=%s threshold=%.2f", model_path, threshold)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self._input_size) if (w, h) != self._input_size else frame

        results = self._model.track(
            resized,
            persist=True,
            conf=self._threshold,
            classes=self._class_ids(),
            verbose=False,
        )

        detections: list[Detection] = []
        frame_area = self._input_size[0] * self._input_size[1]

        if results[0].boxes.id is None:
            return detections

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

            det = Detection(
                track_id=int(track_id),
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                area_fraction=area_frac,
            )

            if self._is_stationary(det.track_id, det.center):
                continue

            detections.append(det)

        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _class_ids(self) -> list[int]:
        return [
            idx
            for idx, name in self._model.names.items()
            if name in _TRACKED_CLASSES
        ]

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
