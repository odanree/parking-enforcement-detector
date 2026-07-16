"""Detection strategy — swappable per-mode configuration for the pipeline.

Thin strategy pattern: the pipeline loop stays a single function; the strategy
supplies the ~6 knobs that actually differ between detection modes:

  • tracked_classes      — which YOLO COCO classes the detector should track
  • zone_key             — which entry in the zones config file to use
  • zones_config_path    — which YAML file holds the zones
  • use_parking_gates    — enable the pose/wand/chalking/classifier/RAG stack
  • vlm_prompts()        — user + system prompts for the primary/confirm VLM
  • alert_category       — event type string for Notifier + state.record_alert
  • on_positive(event)   — post-detection hook (e.g. slew a secondary PTZ camera)

Parking mode's ``on_positive`` is a no-op (Null Object pattern); rodent mode's
issues a PTZ command to the secondary camera via src.stream.slew.

Rodent mode intentionally skips the parking-only stages (pose priors, chalk-
wand gate, ChalkingAnalyzer buffer + cooldown, person-type classifier, RAG
neighbours, chalker YOLO fast-path). Those exist to reduce cost on a busy
street feed; the rodent site has none of them.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class PositiveEvent:
    """What the strategy sees when a positive is confirmed."""
    track_id: int
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2 in detection-resolution coords
    frame_width: int
    frame_height: int
    confidence: float


class DetectionStrategy(Protocol):
    name: str
    tracked_classes: set[str]
    zone_key: str
    zones_config_path: str
    use_parking_gates: bool
    alert_category: str

    def vlm_prompts(self) -> "tuple[str | None, str | None]": ...
    def on_positive(self, event: PositiveEvent) -> None: ...


# ── Parking (existing behaviour) ─────────────────────────────────────────────

class ParkingEnforcementStrategy:
    name = "parking"
    tracked_classes = {"person", "truck", "motorcycle", "car", "chalker"}
    zone_key = "street_zone"
    zones_config_path = "config/detection.yaml"
    use_parking_gates = True
    alert_category = "chalking"

    def vlm_prompts(self) -> "tuple[str | None, str | None]":
        # None → VLMAnalyzer keeps its module-level defaults (chalking prompts).
        return (None, None)

    def on_positive(self, event: PositiveEvent) -> None:
        # Null object: parking alerts are handled entirely by the existing
        # Notifier + state.record_alert path in pipeline.run().
        return None


# ── Rodent (new site) ────────────────────────────────────────────────────────

_RODENT_SYSTEM_PROMPT = (
    "You are a vision analyst examining short-range surveillance frames looking "
    "for rodents (rats, mice, small rodents). Report only what is visually "
    "present. Always respond with valid JSON only, no markdown, no commentary."
)

_RODENT_USER_PROMPT = (
    "Analyze this frame for a rodent (rat, mouse, small rodent).\n"
    "Return JSON with these exact keys:\n"
    "{\n"
    '  "rodent_detected":     <bool>,   // a live rodent is visibly present\n'
    '  "confidence":          <0..1>,   // your visual certainty\n'
    '  "description":         <one short sentence: species best-guess, location, motion>\n'
    "}\n"
    "Notes:\n"
    "- Cats, squirrels, birds, insects are NOT rodents — return false.\n"
    "- Shadows, leaves, drifting debris, and moving fabric are NOT rodents.\n"
    "- A small elongated body with a visible tail is the strongest signal.\n"
    "- If ambiguous, prefer false with a description explaining the ambiguity.\n"
)


class RodentStrategy:
    name = "rodent"
    # YOLO COCO does not include rat/mouse. Track cat + dog only to route around
    # obvious non-rodent movers (they trigger motion; letting them through wastes
    # a VLM call). Everything else falls to motion → VLM classification.
    tracked_classes = {"cat", "dog"}
    zone_key = "yard_zone"
    zones_config_path = "config/rodent.yaml"
    use_parking_gates = False
    alert_category = "rodent"

    def __init__(self) -> None:
        self._slew = None  # lazily imported so the module has no PTZ dependency at import time

    def vlm_prompts(self) -> "tuple[str | None, str | None]":
        return (_RODENT_USER_PROMPT, _RODENT_SYSTEM_PROMPT)

    def on_positive(self, event: PositiveEvent) -> None:
        # Slew the secondary PTZ camera to the zone containing the detection.
        # Guarded by RODENT_SLEW_ENABLED so dev/replay runs don't move real hardware.
        if os.getenv("RODENT_SLEW_ENABLED", "false").lower() != "true":
            return
        try:
            if self._slew is None:
                from src.stream.slew import get_dispatcher
                self._slew = get_dispatcher()
            self._slew.slew_to_bbox(
                bbox=event.bbox,
                frame_width=event.frame_width,
                frame_height=event.frame_height,
                event_key=("rodent", event.track_id),
            )
        except Exception:
            logger.exception("Rodent slew failed for track=%d", event.track_id)


# ── Factory ──────────────────────────────────────────────────────────────────

def strategy_from_env() -> DetectionStrategy:
    """Return the strategy selected by DETECTION_MODE (default: parking)."""
    mode = os.getenv("DETECTION_MODE", "parking").lower().strip()
    if mode == "rodent":
        logger.info("DetectionStrategy: rodent")
        return RodentStrategy()
    logger.info("DetectionStrategy: parking")
    return ParkingEnforcementStrategy()
