"""Fisheye lens undistortion using pre-computed remap tables.

Pre-computes cv2.fisheye remap maps at startup so per-frame cost is a single
cv2.remap call (~1 ms on CPU for 1280×720).

Calibration values live in config/camera_calibration.yaml.  The defaults are
placeholders — replace fx/fy/cx/cy and distortion_coefficients with real
values from scripts/calibrate_camera.py once you have a checkerboard clip.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class FrameUndistorter:
    def __init__(self, config_path: str = "config/camera_calibration.yaml") -> None:
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        cm = cfg["camera_matrix"]
        K = np.array(
            [[cm["fx"], 0.0, cm["cx"]],
             [0.0, cm["fy"], cm["cy"]],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        D = np.array(cfg["distortion_coefficients"], dtype=np.float64).reshape(4, 1)

        w = int(cfg.get("output_width", 1280))
        h = int(cfg.get("output_height", 720))
        alpha = float(cfg.get("alpha", 0.3))

        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=alpha
        )

        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2
        )
        self._out_size = (w, h)
        logger.info(
            "Undistorter ready — output=%dx%d alpha=%.1f  "
            "(placeholder coefficients until calibration)",
            w, h, alpha,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(
            frame, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
