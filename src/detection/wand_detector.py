"""Chalk-wand detector — classical CV, no ML.

Parking-enforcement officers chalk tires with a long, rigid, near-white wand
held in one hand and angled to the ground. In native-resolution camera footage
this is a distinctive geometric feature: a long, near-straight, bright,
*diagonal* line segment whose upper end sits near the person's hand/lower torso
and whose lower end reaches toward the ground.

This detector keys on that geometry. It is deliberately resolution-dependent:
the wand is ~0.5–0.8× the person's height in pixels, so it is only reliable on
a crop taken at (or near) native sensor resolution — NOT on a 1280×720-scaled
full frame where the wand collapses to a couple of pixels. Feed it a tight
native-res crop around the person.

It is intended as a cheap, high-precision PRE-GATE in front of the VLM: when a
wand is present the candidate is escalated; when absent the VLM call is skipped.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WandSignal:
    present: bool = False
    length_px: float = 0.0
    length_ratio: float = 0.0          # length / person bbox height
    angle_deg: float = 0.0             # 0=horizontal, 90=vertical (folded to 0–90)
    confidence: float = 0.0            # 0–1 heuristic score
    coords: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2 in crop coords

    def as_dict(self) -> dict:
        return {
            "present":      self.present,
            "length_px":    round(self.length_px, 1),
            "length_ratio": round(self.length_ratio, 3),
            "angle_deg":    round(self.angle_deg, 1),
            "confidence":   round(self.confidence, 3),
        }


class WandDetector:
    def __init__(
        self,
        white_v_min: int = 165,        # HSV value floor for "bright"
        white_s_max: int = 70,         # HSV saturation ceiling for "near-white"
        min_angle_deg: float = 18.0,   # exclude near-horizontal (curb/car edges)
        max_angle_deg: float = 72.0,   # exclude near-vertical (poles/fence rails)
        min_length_ratio: float = 0.45,  # wand length relative to person bbox height
        hough_threshold: int = 35,
        max_line_gap: int = 18,
        max_width_px: float = 12.0,       # a wand is THIN; reject fat bright blobs (shirts)
        max_width_ratio: float = 0.18,    # also cap width relative to person width
        require_dark_torso: bool = False, # reject bright-torso subjects (e.g. white shirts)
        dark_torso_v_max: int = 150,      # mean torso V above this = "not a dark uniform"
        min_ground_ratio: float = 0.7,    # line's lower end must reach this far down the person
    ) -> None:
        self._v_min = white_v_min
        self._s_max = white_s_max
        self._min_ang = min_angle_deg
        self._max_ang = max_angle_deg
        self._min_ratio = min_length_ratio
        self._hough_thresh = hough_threshold
        self._max_gap = max_line_gap
        self._max_width_px = max_width_px
        self._max_width_ratio = max_width_ratio
        self._require_dark_torso = require_dark_torso
        self._dark_torso_v_max = dark_torso_v_max
        self._min_ground_ratio = min_ground_ratio

    def detect(
        self,
        crop_bgr: np.ndarray,
        person_bbox_in_crop: tuple[int, int, int, int],
        motion_mask: np.ndarray | None = None,
    ) -> WandSignal:
        """Detect a chalk wand in a native-resolution person crop.

        crop_bgr            : BGR image, a tight native-res crop around the person
                              with padding for the tool (recommend ~1.5–2× bbox).
        person_bbox_in_crop : (x1,y1,x2,y2) of the person inside `crop_bgr`.
        motion_mask         : optional uint8 foreground mask (same H×W as crop).
                              When supplied, the bright-pixel search is restricted
                              to moving foreground — this removes the dominant
                              false-positive source (static white car edges, fence
                              rails, parking lines, foliage highlights). A carried
                              wand moves with the officer and survives the gate.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return WandSignal()
        h, w = crop_bgr.shape[:2]
        px1, py1, px2, py2 = person_bbox_in_crop
        person_h = max(1, py2 - py1)
        min_len = self._min_ratio * person_h

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Dark-torso gate: the chalking officer wears a dark uniform; a
        # white-shirt resident produces a bright torso whose lower edge mimics a
        # diagonal wand. Reject when the torso (upper 60% of the person bbox) is
        # too bright. Opt-in because it assumes a dark-uniform target.
        if self._require_dark_torso:
            ty2 = py1 + int(0.6 * person_h)
            torso = hsv[max(0, py1):max(py1 + 1, ty2), max(0, px1):max(px1 + 1, px2), 2]
            if torso.size and float(torso.mean()) > self._dark_torso_v_max:
                return WandSignal()

        white = cv2.inRange(hsv, (0, 0, self._v_min), (180, self._s_max, 255))
        if motion_mask is not None and motion_mask.shape[:2] == white.shape[:2]:
            # Dilate motion slightly so a thin moving wand isn't clipped at edges.
            mm = cv2.dilate(motion_mask, np.ones((7, 7), np.uint8), iterations=1)
            white = cv2.bitwise_and(white, white, mask=mm)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Wand thickness allowance (px). Floor for tiny/distant subjects, scaled
        # up for large/near ones.
        corridor_w = max(self._max_width_px, self._max_width_ratio * max(1, px2 - px1))

        lines = cv2.HoughLinesP(
            white, 1, np.pi / 180,
            threshold=self._hough_thresh,
            minLineLength=int(min_len),
            maxLineGap=self._max_gap,
        )
        if lines is None:
            return WandSignal()

        # The wand's upper end should be near the person (hand/torso); its lower
        # end reaches below. Score candidates and keep the best.
        best: WandSignal | None = None
        for ln in lines:
            ax, ay, bx, by = ln[0]
            length = math.hypot(bx - ax, by - ay)
            if length < min_len:
                continue
            ang = abs(math.degrees(math.atan2(by - ay, bx - ax)))
            ang = min(ang, 180 - ang)   # fold to 0–90
            if not (self._min_ang <= ang <= self._max_ang):
                continue

            # Lower endpoint = larger y. Wand should descend toward the ground.
            top = (ax, ay) if ay < by else (bx, by)
            bot = (ax, ay) if ay >= by else (bx, by)

            # Upper end must be near the person horizontally (hand region),
            # within one bbox-width of the person column.
            person_cx = (px1 + px2) / 2
            person_w = max(1, px2 - px1)
            if abs(top[0] - person_cx) > 1.5 * person_w:
                continue
            # Upper end should be around lower-torso/hand height or below
            # (not up at the head). Require top above the lower edge but below
            # the person's vertical midpoint.
            if top[1] < py1 + 0.25 * person_h:
                continue

            # REACHES THE GROUND — the decisive discriminator vs a white-shirt
            # edge. A carried chalk wand descends past the legs toward the
            # ground; its lower endpoint sits near or below the feet. A shirt's
            # lower hem sits up at mid-torso. Require the bottom endpoint in the
            # lower part of the person (or below, into the pad region).
            if bot[1] < py1 + self._min_ground_ratio * person_h:
                continue

            # THINNESS — corridor spillover test. A wand's white pixels stay
            # within a narrow band around the line; a white-shirt edge spills
            # into the shirt body on one side. Compare white inside a narrow
            # corridor vs a 3× wider one.
            narrow = np.zeros(white.shape, np.uint8)
            wide = np.zeros(white.shape, np.uint8)
            cv2.line(narrow, (int(ax), int(ay)), (int(bx), int(by)), 255, max(2, int(corridor_w)))
            cv2.line(wide, (int(ax), int(ay)), (int(bx), int(by)), 255, max(4, int(corridor_w * 3)))
            w_narrow = int(cv2.countNonZero(cv2.bitwise_and(white, narrow)))
            w_wide = int(cv2.countNonZero(cv2.bitwise_and(white, wide)))
            spill = (w_wide - w_narrow) / max(1, w_narrow)
            if spill > 1.2:        # >120% extra white outside the thin band → blob, not wand
                continue

            ratio = length / person_h
            # Confidence: longer + more diagonal (closer to ~35°) scores higher.
            ang_score = 1.0 - abs(ang - 35.0) / 55.0    # peak near 35°
            conf = max(0.0, min(1.0, 0.5 * min(1.0, ratio / 0.8) + 0.5 * ang_score))
            cand = WandSignal(
                present=True, length_px=length, length_ratio=ratio,
                angle_deg=ang, confidence=conf,
                coords=(int(ax), int(ay), int(bx), int(by)),
            )
            if best is None or cand.confidence > best.confidence:
                best = cand

        return best or WandSignal()


@dataclass
class _TrackState:
    hits: deque = field(default_factory=lambda: deque(maxlen=12))      # bool wand-present per recent eval
    confidences: deque = field(default_factory=lambda: deque(maxlen=12))
    centers: deque = field(default_factory=lambda: deque(maxlen=12))   # person center per recent eval
    confirmed: bool = False


class WandTracker:
    """Temporal confirmation layer over WandDetector.

    A single bright diagonal in one frame is not enough — static scene
    structure (car trim, fence rails, parking lines) produces those constantly.
    A real chalk wand (a) recurs across consecutive evaluations and (b) moves
    with the person. This tracker confirms a wand for a track only when it is
    present in at least ``min_hits`` of the last ``window`` evaluations AND the
    person has actually moved over that window (a static FP on a parked-looking
    blob is rejected).
    """

    def __init__(
        self,
        window: int = 8,
        min_hits: int = 4,
        min_track_motion_px: float = 12.0,
        evict_after_misses: int = 30,
    ) -> None:
        self._window = window
        self._min_hits = min_hits
        self._min_motion = min_track_motion_px
        self._evict_after = evict_after_misses
        self._tracks: dict[int, _TrackState] = {}
        self._misses: dict[int, int] = {}

    def update(self, track_id: int, signal: WandSignal, person_center: tuple[int, int]) -> bool:
        st = self._tracks.setdefault(track_id, _TrackState())
        st.hits.append(bool(signal.present))
        st.confidences.append(signal.confidence if signal.present else 0.0)
        st.centers.append(person_center)
        self._misses[track_id] = 0

        recent_hits = sum(1 for h in list(st.hits)[-self._window:] if h)
        # Track motion over the window — reject wands "carried" by a static blob.
        cs = list(st.centers)[-self._window:]
        motion = 0.0
        if len(cs) >= 2:
            motion = max(
                math.hypot(cs[i][0] - cs[0][0], cs[i][1] - cs[0][1])
                for i in range(1, len(cs))
            )
        st.confirmed = recent_hits >= self._min_hits and motion >= self._min_motion
        return st.confirmed

    def confidence(self, track_id: int) -> float:
        st = self._tracks.get(track_id)
        if not st or not st.confidences:
            return 0.0
        vals = [c for c in list(st.confidences)[-self._window:] if c > 0]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def tick_absent(self, active_ids: set[int]) -> None:
        """Call once per frame with the currently-tracked ids to evict gone tracks."""
        for tid in list(self._tracks):
            if tid not in active_ids:
                self._misses[tid] = self._misses.get(tid, 0) + 1
                if self._misses[tid] >= self._evict_after:
                    self._tracks.pop(tid, None)
                    self._misses.pop(tid, None)
