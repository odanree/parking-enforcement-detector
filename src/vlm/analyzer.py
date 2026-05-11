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
import re

import anthropic
import httpx

logger = logging.getLogger(__name__)

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

# Lenient prompt — used by the first-pass stage (Ollama).
# Goal: catch any person who MIGHT be chalking.
# DO NOT ask the model to classify the environment (street vs driveway vs lot) —
# the camera angle makes curb-parked vehicles look like lot/driveway vehicles and
# causes false negatives. Focus only on person + vehicle + wheel-level proximity.
_LENIENT_USER_PROMPT = (
    'Analyze this street camera image for people near vehicles.\n'
    'CAMERA NOTE: This is a fixed overhead/angled camera. Do NOT reject based on whether '
    'the background looks like a street, driveway, or parking lot — all look similar at this angle.\n'
    'Set chalking_detected: true if ALL of these apply:\n'
    '  • A person is visible in the frame, AND\n'
    '  • A vehicle is immediately adjacent to the person (within arm\'s reach).\n'
    'Set chalking_detected: false ONLY if:\n'
    '  • No person is visible.\n'
    '  • No vehicle is visible near the person.\n'
    '  • The person is clearly inside a vehicle as driver or passenger.\n'
    '  • The person is far from any vehicle (more than 2–3 metres away).\n'
    'This is a people-detection pass only — do NOT assess chalk tools, crouching posture, '
    'or any specific enforcement activity. Simply confirm: is a person present next to a vehicle?\n'
    'Output only a JSON object with: '
    '{ "chalking_detected": boolean, "sweeper_detected": false, '
    '"pe_vehicle_detected": false, "confidence": float, "description": string }'
)

_LENIENT_SYSTEM_PROMPT = (
    "You are a people-detection filter for an overhead street camera. "
    "Your only job is to confirm whether a person is present immediately next to a vehicle. "
    "Do not assess what the person is doing, whether they are chalking, or any enforcement activity — "
    "that is handled by a later stage. "
    "Flag chalking_detected: true if a person and a nearby vehicle are both visible. "
    "Flag false only if no person is visible, no nearby vehicle, or the person is clearly inside a vehicle. "
    "Respond with valid JSON only — no markdown, no commentary."
)

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
}

_FALLBACK = {
    "chalking_detected": False,
    "sweeper_detected": False,
    "pe_vehicle_detected": False,
    "confidence": 0.0,
    "description": "VLM analysis failed",
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
    ) -> None:
        self._backend = backend
        self._ollama_url = ollama_url.rstrip("/")
        self._ollama_model = ollama_model
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

    def set_prompts(self, user_prompt: str | None = None, system_prompt: str | None = None) -> None:
        if user_prompt is not None:
            self._user_prompt = user_prompt
        if system_prompt is not None:
            self._system_prompt = system_prompt
        logger.info("VLM prompts updated (backend=%s)", self._backend)

    def get_prompts(self) -> dict:
        return {"user_prompt": self._user_prompt, "system_prompt": self._system_prompt}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes | list[bytes], kind: str = "") -> dict:
        """Send one or more JPEG frames to the VLM and return the parsed JSON result.

        When a list is passed the frames are treated as a chronological sequence
        (oldest first) so the model can use earlier context to disambiguate.
        """
        if self._backend == "mock":
            return _MOCK_RESULTS.get(kind, _MOCK_RESULTS["pe_vehicle"])
        frames = image_bytes if isinstance(image_bytes, list) else [image_bytes]
        try:
            if self._backend == "claude":
                return self._analyze_claude(frames)
            return self._analyze_ollama(frames)
        except Exception:
            logger.exception("VLM analysis error")
            return _FALLBACK.copy()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _analyze_claude(self, frames: list[bytes]) -> dict:
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
        parsed = _parse_json(response.content[0].text)
        parsed["_usage"] = {
            "model": self._claude_model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        }
        return parsed

    def _analyze_ollama(self, frames: list[bytes]) -> dict:
        payload = {
            "model": self._ollama_model,
            "prompt": f"{self._system_prompt}\n\n{self._user_prompt}",
            "images": [base64.standard_b64encode(fb).decode() for fb in frames],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        logger.debug("Ollama request: %d frame(s)", len(frames))

        resp = httpx.post(
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return _parse_json(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _normalize(data: dict) -> dict:
    return {
        "chalking_detected":  bool(data.get("chalking_detected", False)),
        "sweeper_detected":   bool(data.get("sweeper_detected", False)),
        "pe_vehicle_detected": bool(data.get("pe_vehicle_detected", False)),
        "confidence":         float(data.get("confidence", 0.0)),
        "description":        str(data.get("description", "")),
    }
