"""Vision-language model client.

Supports three backends:
  • claude  — Anthropic claude-haiku-4-5-20251001 via the API (fast, ~$0.0001/call).
              System prompt is cache-controlled so repeated calls are ~5× cheaper.
  • ollama  — Local LLaVA-style model via Ollama REST API (zero API cost, 8 GB VRAM).
              Use a 4-bit or 8-bit quantised 7 B model to stay under 2 s/frame.
  • mock    — Always returns chalking_detected=True. Use to verify the full alert
              chain (zone → analyzer → notifier → dashboard) without a real VLM.

Returns a dict matching the spec schema:
    {
        "chalking_detected":    bool,
        "sweeper_detected":     bool,
        "pe_vehicle_detected":  bool,
        "confidence":           float,  # 0.0–1.0
        "description":          str,
    }
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re

import anthropic
import httpx

try:
    from langfuse.decorators import langfuse_context, observe
except ImportError:
    def observe(**_kw):
        def _wrap(fn):
            return fn
        return _wrap

    class _LfNoCtx:
        def update_current_observation(self, **_kw): pass
        def update_current_trace(self, **_kw): pass

    langfuse_context = _LfNoCtx()

logger = logging.getLogger(__name__)

_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# Structured-evidence prompt — opt-in via VLM_STRUCTURED_PROMPT=true.
# The model returns discrete observation flags; final classification logic
# lives in Python (apply_chalking_policy) so behaviour is testable and the
# prompt stays short enough for Haiku-class models to follow reliably.
_STRUCTURED_SYSTEM_PROMPT = (
    "You are a vision analyst examining overhead street-camera frames. "
    "Report ONLY what is visually present — do NOT reason about whether it is "
    "chalking. A separate policy layer combines your flags into a decision. "
    "Always respond with valid JSON only, no markdown, no commentary."
)

_STRUCTURED_USER_PROMPT = (
    "Return JSON with these exact keys:\n"
    "{\n"
    '  "tool_visible": <bool>,           // a long thin object held IN a hand pointing toward ground/wheel\n'
    '  "tool_confidence": <0..1>,        // your visual certainty for tool_visible\n'
    '  "crouching_or_bending": <bool>,   // person\'s torso is angled downward OR they are squatting/kneeling\n'
    '  "close_to_rear_wheel": <bool>,    // within ~1 person-width of a parked vehicle\'s rear wheel area\n'
    '  "trunk_or_hatch_open": <bool>,    // any vehicle near the person has a raised trunk, hatch, or open door\n'
    '  "movement_pattern": <"rear_to_front"|"front_to_back"|"stationary"|"leaving_vehicle"|"unclear">,\n'
    '  "notes": <one short sentence describing the scene>,\n'
    '  "observed_confidence": <0..1>     // your overall certainty in the above observations\n'
    "}\n"
    "Notes:\n"
    "- Do NOT compute chalking_detected; leave that to the policy layer.\n"
    "- Backpacks, shoulder bags, shadows, arms, and clothing do NOT count as tools.\n"
    "- 'leaving_vehicle' is when the person walks away from a vehicle's door/trunk area.\n"
    "- 'stationary' is when the person remains in roughly one position across frames.\n"
)

# Strict prompt — used by the confirm stage (Claude/Sonnet).
_USER_PROMPT = (
    'Analyze this street camera crop (overhead/wide-angle, person appears small) '
    'for parking-enforcement events:\n'
    '1. CHALKING — two-step evaluation, follow in order:\n'
    '   STEP 1 (EXCLUSION CHECK): Look at every vehicle near the person. '
    'Does ANY vehicle have an open trunk lid, raised hatch, or open door — or is the trunk/hatch '
    'state AMBIGUOUS (a raised or unclear element that could be an open trunk)? '
    'If YES or AMBIGUOUS → err on the side of caution, set chalking_detected: false and skip STEP 2. '
    'If the trunk/hatch/doors are CLEARLY closed → the exclusion does NOT apply, PROCEED to STEP 2.\n'
    '   STEP 2 (EVIDENCE CHECK — run this when STEP 1 found no open trunks): '
    'Set chalking_detected: true if EITHER condition below is true:\n'
    '   CONDITION A — TOOL: A long thin object (rod, pole, wand, chalk stick) that is visibly '
    'held IN the person\'s hand and extends downward or laterally toward the ground or vehicle. '
    'The object must originate from the hand — do NOT flag backpacks, shoulder bags, items worn '
    'on the back, or anything not clearly in the hand. '
    'Do NOT flag a person just because their arm, shadow, or clothing creates a thin line. '
    'The object must be a distinct item separate from the body, held in the hand, pointing '
    'downward or toward the wheel/ground level.\n'
    '   CONDITION B — PE OFFICER MOVEMENT PATTERN: You are reviewing multiple frames. '
    'Flag chalking_detected: true ONLY if the person exhibits one of these PE officer behaviors:\n'
    '   B1 (STRONGEST SIGNAL) — REAR-TO-FRONT PASS: The person is walking from the rear of the '
    'vehicle toward the front (back-to-front direction) while staying close to the vehicle at '
    'wheel/door level. PE officers mark efficiently while walking — they do not need to stop long.\n'
    '   B2 — BRIEF STATIONARY DWELL WITH CONFIRMED REAR APPROACH: The person stands or crouches '
    'at the rear wheel position for a SHORT time, AND ALL THREE of the following must be true:\n'
    '     (i)  Approach direction is CLEARLY from the rear or side — if it is ambiguous or '
    'cannot be determined from the frames, B2 does NOT apply.\n'
    '     (ii) The dwell is brief — a prolonged stationary presence is a vehicle owner, not a PE officer.\n'
    '     (iii) There is at least partial tool evidence — Condition A is at least tentatively met '
    '(some object appears to extend from the hand). B2 with zero tool evidence → chalking_detected: false.\n'
    '   B3 — FRONT-TO-BACK STOP WITH CLEAR PIVOT: The person walked front-to-back, stopped at '
    'the rear wheel, and visibly reversed direction (now moving rear-to-front). Just stopping '
    'at the rear after a front-to-back walk is NOT B3 — the reversal must be visible.\n'
    '   EXCLUSIONS (set chalking_detected: false for any of these):\n'
    '   • Person walked front-to-back and continued past the rear wheel toward the trunk.\n'
    '   • Person walked front-to-back and stopped/dwelled at the rear wheel — this is a '
    'vehicle owner pausing before accessing their trunk, NOT a PE officer.\n'
    '   • Approach direction is ambiguous and no tool is visible.\n'
    '   IMPORTANT: CONDITION A alone is sufficient (tool clearly visible in hand → flag true, '
    'overrides all direction rules). '
    'B1 alone (clear rear-to-front direction, no tool visible) → may detect but cap confidence at 0.62. '
    'B2 requires confirmed rear/side approach AND partial tool evidence — either missing → false. '
    'A "possible" or "tentative" tool does NOT satisfy Condition A standalone but does satisfy '
    'the partial-tool requirement for B2. '
    'If BOTH A (confirmed) and B are present, confidence should be very high.\n'
    'Output only a JSON object with: '
    '{ "chalking_detected": boolean, "sweeper_detected": false, '
    '"pe_vehicle_detected": false, "confidence": float, "description": string }'
)

_SYSTEM_PROMPT = (
    "You are a parking-enforcement detection AI analyzing overhead street camera footage. "
    "For chalking: first check if any vehicle near the person has an open trunk, hatch, or door, "
    "OR if the trunk/hatch state is ambiguous or unclear — if yes or ambiguous, return chalking_detected: false. "
    "Only proceed if trunks and doors are CLEARLY closed. "
    "Flag chalking true if: (A) a long thin object is clearly held IN the person's hand, extending downward "
    "or toward the ground — backpacks, shoulder bags, arms, shadows, and worn items do NOT qualify; OR "
    "(B) the person exhibits PE officer movement near the rear wheel — "
    "B1: walking rear-to-front along the vehicle (strongest signal; cap confidence at 0.62 if no tool visible); "
    "B2: brief dwell at rear wheel with CONFIRMED rear/side approach AND at least partial tool evidence — "
    "ambiguous approach direction → B2 does not apply; zero tool evidence → B2 does not apply; "
    "B3: walked front-to-back but visibly reversed direction (pivot) at the rear wheel. "
    "EXCLUSIONS: front-to-back walk that continues past → false; front-to-back walk that stops without reversing → false; "
    "ambiguous approach + no tool → false. "
    "CONDITION A (tool confirmed) alone is always sufficient. "
    "A 'possible' or 'tentative' tool satisfies B2's partial-tool requirement but NOT standalone Condition A. "
    "Respond with valid JSON only — no markdown, no commentary."
)

# First-pass prompt — used by the primary stage.
# Goal: confirm the YOLO bbox is actually a person, nothing else.
# No vehicle context, no chalking assessment, no movement patterns.
# A later confirm stage handles all of that.
_LENIENT_USER_PROMPT = (
    'A YOLO object detector flagged a person in this image. '
    'Confirm whether a human being is clearly visible.\n'
    'Set chalking_detected: true if a person is present in the frame.\n'
    'Set chalking_detected: false ONLY if the detection is a false alarm — '
    'e.g. a shadow, an animal, a mannequin, or no person at all.\n'
    'Do NOT evaluate vehicles, tools, movement, or anything else — '
    'that is handled by a later stage.\n'
    'Output only a JSON object with: '
    '{ "chalking_detected": boolean, "sweeper_detected": false, '
    '"pe_vehicle_detected": false, "confidence": float, "description": string }\n'
    'The "description" should be one short sentence describing only whether '
    'a person is visible (e.g. "Person visible near left edge of frame." or '
    '"No person detected — appears to be a shadow.").'
)

_LENIENT_SYSTEM_PROMPT = (
    "You are a YOLO detection validator for a street camera. "
    "Your only job: confirm whether the flagged region contains a real human being. "
    "Ignore vehicles, surroundings, and context entirely. "
    "Set chalking_detected: true if a person is present, false if it is a false alarm. "
    "Respond with valid JSON only — no markdown, no commentary."
)

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a person-type classifier for an overhead street camera. "
    "Classify the person visible in the image into exactly one of these categories:\n"
    "  pedestrian      — walking past on the sidewalk or street, no interaction with any vehicle\n"
    "  resident        — a person who lives at this residence (entering/leaving the home or gate, "
    "moving around the property, in casual home clothing)\n"
    "  occupant        — getting in or out of a parked vehicle, loading/unloading bags or groceries\n"
    "  worker_landscape — gardener, landscaper, or maintenance worker (mowing, trimming, cleaning)\n"
    "  worker_delivery — delivery driver carrying packages (UPS, FedEx, USPS, Amazon, food delivery)\n"
    "  chalker         — parking enforcement officer marking tires or writing a ticket\n"
    "  unknown         — cannot determine from this image\n"
    "Respond with valid JSON only — no markdown, no commentary."
)

_CLASSIFY_USER_PROMPT = (
    "CAMERA NOTE: This is a fixed overhead/angled street camera. Vehicles may appear from above. "
    "Classify the person visible in this image.\n"
    "Consider: posture, carried items, proximity to vehicle, clothing/uniform, direction of movement.\n"
    "Output only JSON: "
    '{ "person_type": "<category>", "confidence": <0.0–1.0>, "description": "<one sentence>" }'
)

_CLASSIFY_FALLBACK = {
    "person_type": "unknown",
    "confidence":  0.0,
    "description": "Classification failed",
}


def _normalize_classify(data: dict) -> dict:
    valid = {"pedestrian", "resident", "occupant", "worker_landscape", "worker_delivery", "chalker", "unknown"}
    pt = str(data.get("person_type", "unknown")).lower().strip()
    if pt not in valid:
        pt = "unknown"
    return {
        "person_type": pt,
        "confidence":  _normalize_confidence(data.get("confidence", 0.0)),
        "description": str(data.get("description", "")),
    }


_MOCK_RESULTS: dict[str, dict] = {
    "chalking": {
        "chalking_detected": True,
        "sweeper_detected": False,
        "pe_vehicle_detected": False,
        "confidence": 0.91,
        "description": (
            "Officer visible crouching near rear tire of parked vehicle. "
            "Posture and proximity to wheel area consistent with tire chalking activity."
        ),
    },
    "sweeper": {
        "chalking_detected": False,
        "sweeper_detected": True,
        "pe_vehicle_detected": False,
        "confidence": 0.88,
        "description": (
            "Street sweeper truck detected in frame. Large rotating side brushes visible "
            "along curb line. Vehicle moving slowly consistent with active sweep operation."
        ),
    },
    "pe_vehicle": {
        "chalking_detected": False,
        "sweeper_detected": False,
        "pe_vehicle_detected": True,
        "confidence": 0.87,
        "description": (
            "White city vehicle with parking enforcement markings stopped adjacent to curb. "
            "Officer visible near driver door. Vehicle has been stationary for several seconds."
        ),
    },
    "rodent": {
        "rodent_detected": True,
        "chalking_detected": False,
        "sweeper_detected": False,
        "pe_vehicle_detected": False,
        "confidence": 0.86,
        "description": (
            "Small elongated body with visible tail moving low along the ground. "
            "Motion pattern consistent with a rat/mouse."
        ),
    },
}

_FALLBACK = {
    "chalking_detected": False,
    "sweeper_detected": False,
    "pe_vehicle_detected": False,
    "confidence": 0.0,
    "description": "VLM analysis failed",
}


_STRUCTURED_MODE = os.getenv("VLM_STRUCTURED_PROMPT", "false").lower() == "true"


def apply_chalking_policy(obs: dict) -> dict:
    """Combine the VLM's structured observations into a chalking verdict.

    Policy precedence (highest to lowest):
      1. trunk_or_hatch_open OR leaving_vehicle → false (exclusion)
      2. tool_visible AND tool_confidence ≥ 0.5 → true (tool gate)
      3. crouching_or_bending AND close_to_rear_wheel → true (posture gate)
      4. movement_pattern == 'rear_to_front' AND close_to_rear_wheel → true
         (officer walking pattern; conf capped at 0.65 if no posture/tool)
      5. otherwise → false

    Returns a dict in the canonical pipeline shape with a synthesized
    ``description`` for the kanban view.
    """
    trunk      = bool(obs.get("trunk_or_hatch_open", False))
    leaving    = obs.get("movement_pattern") == "leaving_vehicle"
    tool       = bool(obs.get("tool_visible", False))
    tool_c     = float(obs.get("tool_confidence", 0.0))
    crouch     = bool(obs.get("crouching_or_bending", False))
    close      = bool(obs.get("close_to_rear_wheel", False))
    pattern    = str(obs.get("movement_pattern", "unclear"))
    obs_conf   = float(obs.get("observed_confidence", 0.5))
    notes      = str(obs.get("notes", ""))

    detected = False
    reason = ""
    confidence = 0.0

    if trunk or leaving:
        reason = "trunk/door/hatch open or person leaving vehicle"
    elif tool and tool_c >= 0.5:
        detected, reason, confidence = True, "tool visible in hand", min(1.0, max(tool_c, obs_conf))
    elif crouch and close:
        detected, reason, confidence = True, "crouch + adjacent to rear wheel", obs_conf
    elif pattern == "rear_to_front" and close:
        detected, reason, confidence = True, "rear-to-front officer pattern near wheel", min(0.65, obs_conf)
    else:
        reason = "no qualifying evidence"

    desc = f"[policy] {reason}. {notes}".strip()
    return {
        "chalking_detected":  detected,
        "sweeper_detected":   False,
        "pe_vehicle_detected": False,
        "confidence":         _normalize_confidence(confidence) if detected else _normalize_confidence(obs_conf),
        "description":        desc,
        "_observations":      obs,  # carried through trace for debugging
    }


class VLMAnalyzer:
    def __init__(
        self,
        backend: str = "claude",
        claude_model: str = "claude-haiku-4-5-20251001",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llava:7b-v1.6-mistral-q4_K_M",
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        structured: bool | None = None,
    ) -> None:
        self._backend = backend
        self._ollama_url = ollama_url.rstrip("/")
        self._ollama_model = ollama_model
        self._structured = _STRUCTURED_MODE if structured is None else structured
        if self._structured:
            self._user_prompt = user_prompt or _STRUCTURED_USER_PROMPT
            self._system_prompt = system_prompt or _STRUCTURED_SYSTEM_PROMPT
        else:
            self._user_prompt = user_prompt or _USER_PROMPT
            self._system_prompt = system_prompt or _SYSTEM_PROMPT

        if backend == "claude":
            self._claude = anthropic.Anthropic()
            self._claude_model = claude_model
            logger.info("VLM backend: Claude (%s)", claude_model)
        elif backend == "mock":
            self._claude = None  # type: ignore[assignment]
            logger.warning("VLM backend: MOCK — all detections will be forced True")
        else:
            self._claude = None  # type: ignore[assignment]
            logger.info("VLM backend: Ollama (%s @ %s)", ollama_model, ollama_url)

    @property
    def model_name(self) -> str:
        """Specific model identifier (e.g. 'gemma4:e4b', 'claude-sonnet-4-6', 'mock')."""
        if self._backend == "claude":
            return self._claude_model
        if self._backend == "mock":
            return "mock"
        return self._ollama_model

    def set_prompts(self, user_prompt: str | None = None, system_prompt: str | None = None) -> None:
        if user_prompt is not None:
            self._user_prompt = user_prompt
        if system_prompt is not None:
            self._system_prompt = system_prompt
        logger.info("VLM prompts updated (backend=%s)", self._backend)

    def get_prompts(self) -> dict:
        return {"user_prompt": self._user_prompt, "system_prompt": self._system_prompt}

    # ── Public API ────────────────────────────────────────────────────────────

    @observe(name="vlm.classify_person", capture_input=False)
    def classify_person(self, image_bytes: bytes | list[bytes]) -> dict:
        """Classify the person in the image into a person_type category.

        Returns: { person_type, confidence, description }
        person_type one of: pedestrian, occupant, worker_landscape, worker_delivery, chalker, unknown
        """
        frames_count = len(image_bytes) if isinstance(image_bytes, list) else 1
        langfuse_context.update_current_trace(
            name="vlm.classify_person",
            metadata={"backend": self._backend, "model": self.model_name, "frames": frames_count},
        )
        if self._backend == "mock":
            return {"person_type": "pedestrian", "confidence": 0.9, "description": "Mock classification."}
        frames = image_bytes if isinstance(image_bytes, list) else [image_bytes]
        try:
            raw_text = self._raw_response(frames, _CLASSIFY_SYSTEM_PROMPT, _CLASSIFY_USER_PROMPT)
            parsed   = _parse_json_raw(raw_text)
            return _normalize_classify(parsed)
        except Exception:
            logger.exception("classify_person error")
            return _CLASSIFY_FALLBACK.copy()

    @observe(name="vlm.analyze", capture_input=False)
    def analyze(
        self,
        image_bytes: bytes | list[bytes],
        kind: str = "",
        prior_signals: str | None = None,
    ) -> dict:
        """Send one or more JPEG frames to the VLM and return the parsed JSON result.

        When a list is passed the frames are treated as a chronological sequence
        (oldest first) so the model can use earlier context to disambiguate.

        ``prior_signals`` is an optional short string (≤ 200 chars) describing
        deterministic priors from upstream (e.g. pose: crouching, wrist near
        wheel). It is appended to the user prompt as a hint, not as ground
        truth — the VLM still owns the final decision.
        """
        frames_count = len(image_bytes) if isinstance(image_bytes, list) else 1
        langfuse_context.update_current_trace(
            name="vlm.analyze",
            metadata={
                "backend": self._backend,
                "model": self.model_name,
                "kind": kind,
                "structured": self._structured,
                "frames": frames_count,
                "prior_signals": prior_signals,
            },
        )
        if self._backend == "mock":
            return _MOCK_RESULTS.get(kind, _MOCK_RESULTS["pe_vehicle"])
        frames = image_bytes if isinstance(image_bytes, list) else [image_bytes]
        try:
            if self._backend == "claude":
                raw = self._analyze_claude(frames, prior_signals=prior_signals)
            else:
                raw = self._analyze_ollama(frames, prior_signals=prior_signals)
            if self._structured:
                obs = raw.get("_obs_raw", {})
                policy_result = apply_chalking_policy(obs)
                if "_usage" in raw:
                    policy_result["_usage"] = raw["_usage"]
                return policy_result
            return raw
        except httpx.TimeoutException:
            logger.warning("VLM timeout (%s): took >60 s", self.model_name)
            fb = _FALLBACK.copy()
            fb["description"] = f"VLM timeout — {self.model_name} took >60 s"
            return fb
        except httpx.ConnectError:
            logger.warning("VLM unreachable (%s @ %s)", self.model_name, self._ollama_url)
            fb = _FALLBACK.copy()
            fb["description"] = f"VLM unreachable — {self._ollama_url}"
            return fb
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("VLM model not found — run: ollama pull %s", self._ollama_model)
                fb = _FALLBACK.copy()
                fb["description"] = f"Model not found — run: ollama pull {self._ollama_model}"
                return fb
            logger.exception("VLM HTTP error (%s)", self.model_name)
            fb = _FALLBACK.copy()
            fb["description"] = f"VLM HTTP {exc.response.status_code} — {self.model_name}"
            return fb
        except Exception as exc:
            logger.exception("VLM analysis error (%s)", self.model_name)
            fb = _FALLBACK.copy()
            fb["description"] = f"VLM error — {type(exc).__name__}: {exc}"
            return fb

    # ── Backends ──────────────────────────────────────────────────────────────

    def _raw_response(self, frames: list[bytes], system_prompt: str, user_prompt: str) -> str:
        """Call the backend with custom prompts and return the raw text response."""
        if self._backend == "claude":
            content: list[dict] = []
            for fb in frames:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(fb).decode(),
                    },
                })
            content.append({"type": "text", "text": user_prompt})
            response = self._claude.messages.create(
                model=self._claude_model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            langfuse_context.update_current_observation(
                model=self._claude_model,
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
                output=response.content[0].text,
            )
            return response.content[0].text
        else:
            payload = {
                "model":   self._ollama_model,
                "prompt":  f"{system_prompt}\n\n{user_prompt}",
                "images":  [base64.standard_b64encode(fb).decode() for fb in frames],
                "stream":  False,
                "options": {"temperature": 0.0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))},
            }
            resp = httpx.post(f"{self._ollama_url}/api/generate", json=payload, timeout=_OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "")

    def _analyze_claude(self, frames: list[bytes], prior_signals: str | None = None) -> dict:
        content: list[dict] = []
        for fb in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(fb).decode(),
                },
            })
        prompt = self._user_prompt if len(frames) <= 1 else (
            'You are shown a wide scene image followed by '
            f'{len(frames) - 1} close-up crop(s) of the same tracked person sampled over ~1 second. '
            'Image 1 is the SCENE — use it to assess vehicle state (open trunks, doors, hatches). '
            'Images 2 onwards are DETAIL crops — evaluate them as a sequence to assess posture, '
            'movement, and any objects the person is carrying. '
            'Apply STEP 1 (trunk/door gate) using the scene image. '
            'Apply STEP 2 across all detail frames in totality: '
            'flag true if a tool is visible in ANY frame, OR if the person is stationary at a rear wheel '
            'across the MAJORITY of frames (sustained presence = chalking; brief pass-by = not chalking). '
            'Do NOT dismiss chalking just because one frame looks ambiguous — judge the full sequence.\n\n'
            + self._user_prompt
        )
        if prior_signals:
            prompt = (
                f"PRIOR SIGNALS (deterministic, from pose estimator): {prior_signals}.\n"
                "Treat these as weak evidence — confirm visually before flagging.\n\n"
                + prompt
            )
        content.append({"type": "text", "text": prompt})

        response = self._claude.messages.create(
            model=self._claude_model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "max_tokens":
            logger.warning("Claude response truncated at max_tokens — increase max_tokens")
        usage = {
            "model": self._claude_model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        }
        langfuse_context.update_current_observation(
            model=self._claude_model,
            usage_details={
                "input": usage["input_tokens"],
                "output": usage["output_tokens"],
                "cache_read_input_tokens": usage["cache_read_input_tokens"],
                "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
            },
            output=response.content[0].text,
        )
        if self._structured:
            obs = _parse_json_raw(response.content[0].text)
            return {"_obs_raw": obs, "_usage": usage}
        parsed = _parse_json(response.content[0].text)
        parsed["_usage"] = usage
        return parsed

    def _analyze_ollama(self, frames: list[bytes], prior_signals: str | None = None) -> dict:
        user = self._user_prompt
        if prior_signals:
            user = (
                f"PRIOR SIGNALS (deterministic, from pose estimator): {prior_signals}.\n"
                "Treat these as weak evidence — confirm visually before flagging.\n\n"
                + user
            )
        payload = {
            "model": self._ollama_model,
            "prompt": f"{self._system_prompt}\n\n{user}",
            "images": [base64.standard_b64encode(fb).decode() for fb in frames],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))},
        }
        logger.debug("Ollama request: %d frame(s)", len(frames))

        resp = httpx.post(
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        if self._structured:
            return {"_obs_raw": _parse_json_raw(raw)}
        return _parse_json(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_raw(text: str) -> dict:
    """Parse JSON from LLM response without any schema normalization."""
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.rfind("{")
    if start != -1:
        depth, end = 0, -1
        in_str, escape = False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False; continue
            if ch == "\\" and in_str:
                escape = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse raw VLM JSON: %r", text[:200])
    return {}


def _parse_json(text: str) -> dict:
    """Extract and parse the JSON object from an LLM response.

    Handles three cases:
      1. Bare JSON (ideal)
      2. JSON wrapped in ```json ... ``` fences (local models)
      3. Prose reasoning followed by a JSON object (Sonnet with step-by-step output)
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()

    # Fast path — the whole response is valid JSON
    try:
        data = json.loads(text)
        return _normalize(data)
    except json.JSONDecodeError:
        pass

    # Slow path — find the last { and brace-match to extract the JSON object.
    # The model writes prose first then the JSON, so the last { is our target.
    start = text.rfind("{")
    if start != -1:
        depth, end = 0, -1
        in_str, escape = False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                data = json.loads(text[start:end])
                return _normalize(data)
            except json.JSONDecodeError:
                pass

    logger.warning("Could not parse VLM JSON: %r", text[:200])
    return _FALLBACK.copy()


def _normalize_confidence(value) -> float:
    """Clamp confidence into [0, 1].

    Qwen2.5VL and several other open VLMs frequently return confidence on a
    0-100 (or 0-10) scale despite the prompt asking for 0.0-1.0. Treat any
    value > 1.5 as a percentage and divide accordingly; clamp out-of-range
    values defensively.
    """
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c or c < 0:                          # NaN / negative
        return 0.0
    if c > 1.5 and c <= 100.0:
        c = c / 100.0
    elif c > 100.0:
        return 1.0
    return min(1.0, c)


def _normalize(data: dict) -> dict:
    return {
        "chalking_detected":  bool(data.get("chalking_detected", False)),
        "sweeper_detected":   bool(data.get("sweeper_detected", False)),
        "pe_vehicle_detected": bool(data.get("pe_vehicle_detected", False)),
        "confidence":         _normalize_confidence(data.get("confidence", 0.0)),
        "description":        str(data.get("description", "")),
    }
