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

# Exact prompt from the spec — kept here so it's easy to tune.
_USER_PROMPT = (
    'Analyze this street camera crop (overhead/wide-angle, person appears small) '
    'for parking-enforcement events:\n'
    '1. CHALKING — two-step evaluation, follow in order:\n'
    '   STEP 1 (EXCLUSION CHECK): Look at every vehicle near the person. '
    'Does ANY vehicle have an open trunk lid, raised hatch, or open door? '
    'If YES → person is loading/unloading, set chalking_detected: false and skip STEP 2. '
    'If NO open trunk/hatch/door is visible → the exclusion does NOT apply, PROCEED to STEP 2.\n'
    '   STEP 2 (EVIDENCE CHECK — run this when STEP 1 found no open trunks): '
    'Set chalking_detected: true if EITHER condition below is true:\n'
    '   CONDITION A — TOOL: A long thin object (rod, pole, wand, chalk stick) that is visibly '
    'held IN the person\'s hand and extends downward or laterally toward the ground or vehicle. '
    'The object must originate from the hand — do NOT flag backpacks, shoulder bags, items worn '
    'on the back, or anything not clearly in the hand. '
    'Do NOT flag a person just because their arm, shadow, or clothing creates a thin line. '
    'The object must be a distinct item separate from the body, held in the hand, pointing '
    'downward or toward the wheel/ground level.\n'
    '   CONDITION B — SUSTAINED WHEEL PRESENCE: You are reviewing multiple frames. '
    'The person is directly beside the rear wheel of a parked vehicle AND has remained '
    'stationary at that same wheel position across the majority of the frames in this sequence. '
    'A brief pass-by or someone who is clearly walking does NOT qualify — '
    'the person must be lingering/stationary at the rear wheel. '
    'Any posture qualifies (standing, slightly bent, crouching) as long as they are not moving away.\n'
    '   IMPORTANT: CONDITION A alone is sufficient (tool visible near hand → flag true, '
    'regardless of wheel proximity or posture). '
    'CONDITION B alone requires the sustained stationary presence described above. '
    'If BOTH conditions are present simultaneously, confidence should be very high.\n'
    'Output only a JSON object with: '
    '{ "chalking_detected": boolean, "sweeper_detected": false, '
    '"pe_vehicle_detected": false, "confidence": float, "description": string }'
)

_SYSTEM_PROMPT = (
    "You are a parking-enforcement detection AI analyzing overhead street camera footage. "
    "For chalking: first check if any vehicle near the person has an open trunk, hatch, or door — "
    "if yes, that person is loading/unloading, return chalking_detected: false. "
    "If NO open trunk/door is visible, the exclusion does NOT apply — proceed to look for evidence. "
    "Flag chalking true if: (A) a long thin object is clearly held IN the person's hand, extending downward "
    "or toward the ground — backpacks, shoulder bags, arms, shadows, and worn items do NOT qualify; OR "
    "(B) the person has remained stationary at the rear wheel of a parked vehicle across the "
    "majority of the provided frames — a PE officer lingers; a car owner briefly passes. "
    "CONDITION A alone is sufficient to flag. CONDITION B requires sustained stationary presence. "
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
    ) -> None:
        self._backend = backend
        self._ollama_url = ollama_url.rstrip("/")
        self._ollama_model = ollama_model

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
        prompt = _USER_PROMPT if len(frames) <= 1 else (
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
            + _USER_PROMPT
        )
        content.append({"type": "text", "text": prompt})

        response = self._claude.messages.create(
            model=self._claude_model,
            max_tokens=256,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )
        return _parse_json(response.content[0].text)

    def _analyze_ollama(self, frames: list[bytes]) -> dict:
        payload = {
            "model": self._ollama_model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{_USER_PROMPT}",
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
