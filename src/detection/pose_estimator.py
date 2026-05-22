"""Pose-estimation module — derives deterministic posture signals for
detected persons so the chalking pipeline does not depend entirely on the
VLM to reason about crouching, wrist-near-ground, or wrist-near-vehicle.

Uses YOLOv8-Pose (COCO 17-keypoint skeleton). The class indices are:
   0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
   11-12 hips, 13-14 knees, 15-16 ankles.

Returned ``PoseSignals`` dataclass exposes:

  is_crouching        : hip below knee (perspective-correct) OR knee bent
  hand_low            : lowest wrist keypoint is in the bottom 25 % of the
                        person's bbox — a "reaching down" posture
  wrist_near_vehicle  : either wrist is within ``contact_px`` of any vehicle
                        bbox bottom edge (i.e. near a wheel)
  confidence          : minimum visibility (kpt-conf) across the keypoints
                        that contributed to the above flags
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices
_NOSE, _L_SHOULDER, _R_SHOULDER = 0, 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

# Minimum keypoint confidence to use a coordinate.
_KPT_MIN_CONF = 0.30


@dataclass
class PoseSignals:
    track_id: int                   # which person this pose belongs to (matched on bbox IoU)
    bbox: tuple[int, int, int, int]
    is_crouching: bool = False
    hand_low: bool = False
    wrist_near_vehicle: bool = False
    knee_angle_deg: float = 180.0   # 180 = straight; <120 = clearly bent
    wrist_y_ratio: float = 0.0      # 0 = at bbox top, 1 = at bbox bottom
    confidence: float = 0.0
    keypoints: list[tuple[float, float, float]] | None = None  # (x, y, conf) per joint

    def as_dict(self) -> dict:
        return {
            "track_id":            self.track_id,
            "is_crouching":        self.is_crouching,
            "hand_low":            self.hand_low,
            "wrist_near_vehicle":  self.wrist_near_vehicle,
            "knee_angle_deg":      round(self.knee_angle_deg, 1),
            "wrist_y_ratio":       round(self.wrist_y_ratio, 3),
            "confidence":          round(self.confidence, 3),
        }

    def summary(self) -> str:
        flags = []
        if self.is_crouching:       flags.append("crouching")
        if self.hand_low:           flags.append("hand-low")
        if self.wrist_near_vehicle: flags.append("wrist-near-vehicle")
        return ",".join(flags) if flags else "neutral-stance"


class PoseEstimator:
    """YOLOv8-pose wrapper that returns derived signals per detected person.

    Loaded lazily — the first call to ``estimate`` triggers the model load.
    Skipping ``PoseEstimator`` instantiation when pose is disabled avoids the
    ~50 MB weight download on machines that do not need it.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-pose.pt",
        contact_px: int = 35,
        knee_angle_threshold_deg: float = 130.0,
    ) -> None:
        from ultralytics import YOLO   # deferred — ultralytics is heavy
        self._model = YOLO(model_path)
        self._contact_px = contact_px
        self._knee_thresh = knee_angle_threshold_deg
        logger.info(
            "PoseEstimator ready (model=%s, contact_px=%d, knee_thresh=%.0f°)",
            model_path, contact_px, knee_angle_threshold_deg,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
        person_dets: "list",
        vehicle_bboxes: "list[tuple[int, int, int, int]]",
    ) -> list[PoseSignals]:
        """Run pose inference and return one PoseSignals per matched person."""
        if not person_dets:
            return []
        # Inference on whole frame is faster than per-bbox crops on GPU and the
        # YOLO-pose tracker maintains its own assignment.
        results = self._model.predict(frame, verbose=False, conf=0.35)
        if not results or results[0].keypoints is None:
            return []

        kp_data = results[0].keypoints.data   # tensor (n_people, 17, 3) — x, y, conf
        boxes = results[0].boxes.xyxy
        if kp_data is None or len(kp_data) == 0:
            return []

        kp_np = kp_data.cpu().numpy()
        bx_np = boxes.cpu().numpy() if boxes is not None else None

        # Match pose detections to incoming person tracks by IoU.
        signals: list[PoseSignals] = []
        used_pose_idx: set[int] = set()
        for det in person_dets:
            if det.class_name != "person":
                continue
            best_idx, best_iou = -1, 0.0
            for j in range(len(kp_np)):
                if j in used_pose_idx or bx_np is None:
                    continue
                iou = _iou(det.bbox, tuple(map(int, bx_np[j])))
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_idx < 0 or best_iou < 0.20:
                # No reliable pose for this person — skip.
                continue
            used_pose_idx.add(best_idx)
            ps = self._derive_signals(det, kp_np[best_idx], vehicle_bboxes)
            signals.append(ps)
        return signals

    # ── Signal derivation ────────────────────────────────────────────────────

    def _derive_signals(
        self,
        det,
        kpts: np.ndarray,
        vehicle_bboxes: list[tuple[int, int, int, int]],
    ) -> PoseSignals:
        # kpts is shape (17, 3): x, y, conf
        x1, y1, x2, y2 = det.bbox
        bbox_h = max(1, y2 - y1)

        def pt(i: int) -> tuple[float, float, float] | None:
            x, y, c = kpts[i]
            if c < _KPT_MIN_CONF:
                return None
            return float(x), float(y), float(c)

        # Hand-low: lowest visible wrist relative to bbox height
        wrist_y_ratio = 0.0
        hand_low = False
        wrist_pts: list[tuple[float, float]] = []
        for wi in (_L_WRIST, _R_WRIST):
            p = pt(wi)
            if p is not None:
                wrist_pts.append((p[0], p[1]))
                ratio = (p[1] - y1) / bbox_h
                if ratio > wrist_y_ratio:
                    wrist_y_ratio = ratio
        if wrist_y_ratio > 0.75:
            hand_low = True

        # Crouch: either hip is below knee, OR knee angle is sharp.
        # Use whichever leg has better visibility.
        knee_angle = 180.0
        crouch_hip_below_knee = False
        for hip_i, knee_i, ankle_i in (
            (_L_HIP, _L_KNEE, _L_ANKLE),
            (_R_HIP, _R_KNEE, _R_ANKLE),
        ):
            hp, kp, ap = pt(hip_i), pt(knee_i), pt(ankle_i)
            if hp and kp and ap:
                ang = _angle_deg((hp[0], hp[1]), (kp[0], kp[1]), (ap[0], ap[1]))
                if ang < knee_angle:
                    knee_angle = ang
            # In a normal standing pose hip-y < knee-y (image coords: y grows down).
            # In a deep crouch the hip can descend close to or below knee level.
            if hp and kp and hp[1] >= kp[1] - 6:
                crouch_hip_below_knee = True
        is_crouching = crouch_hip_below_knee or knee_angle < self._knee_thresh

        # Wrist near vehicle: any wrist keypoint within contact_px of any
        # vehicle bbox edge (bias toward bottom edge — wheels live there).
        wrist_near_vehicle = False
        for wx, wy in wrist_pts:
            for vx1, vy1, vx2, vy2 in vehicle_bboxes:
                # distance from wrist to vehicle bbox (0 if inside)
                dx = max(0, max(vx1, wx) - min(vx2, wx))
                dy = max(0, max(vy1, wy) - min(vy2, wy))
                d  = math.hypot(dx, dy)
                # Prioritise bottom-half of vehicle (where wheels appear).
                bottom_dist = abs(wy - vy2)
                if d <= self._contact_px or bottom_dist <= self._contact_px:
                    wrist_near_vehicle = True
                    break
            if wrist_near_vehicle:
                break

        # Overall confidence: lowest visibility of joints that we actually used.
        used_joints = [
            i for i in (_L_HIP, _R_HIP, _L_KNEE, _R_KNEE, _L_WRIST, _R_WRIST)
            if pt(i) is not None
        ]
        conf = (
            float(min(kpts[i, 2] for i in used_joints))
            if used_joints else 0.0
        )

        return PoseSignals(
            track_id=det.track_id,
            bbox=det.bbox,
            is_crouching=is_crouching,
            hand_low=hand_low,
            wrist_near_vehicle=wrist_near_vehicle,
            knee_angle_deg=knee_angle,
            wrist_y_ratio=wrist_y_ratio,
            confidence=conf,
            keypoints=[tuple(kp) for kp in kpts.tolist()],
        )


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _angle_deg(p1: tuple[float, float], pivot: tuple[float, float], p2: tuple[float, float]) -> float:
    """Inner angle at ``pivot`` (degrees, 0–180)."""
    v1 = (p1[0] - pivot[0], p1[1] - pivot[1])
    v2 = (p2[0] - pivot[0], p2[1] - pivot[1])
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))
