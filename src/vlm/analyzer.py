"""Vision-language model client.

Supports two backends:
  • claude  — Anthropic claude-haiku-4-5-20251001 via the API (fast, ~$0.0001/call).
              System prompt is cache-controlled so repeated calls are ~5× cheaper.
  • ollama  — Local LLaVA-style model via Ollama REST API (zero API cost, 8 GB VRAM).
              Use a 4-bit or 8-bit quantised 7 B model to stay under 2 s/frame.

Returns a dict matching the spec schema:
    {
        "chalking_detected": bool,
        "sweeper_detected":  bool,
        "confidence":        float,  # 0.0–1.0
        "description":       str,
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
    'Analyze the person in this frame relative to the vehicle. '
    'Specifically identify if the person is leaning over a tire or holding a '
    'long-reaching tool (chalk stick) toward the ground/tire area. '
    'Separately, identify if any vehicle in the frame has characteristic '
    '"Street Sweeper" features: oversized side brushes, water spray nozzles, '
    'or yellow caution lights. '
    'Output only a JSON object with: '
    '{ "chalking_detected": boolean, "sweeper_detected": boolean, '
    '"confidence": float, "description": string }'
)

_SYSTEM_PROMPT = (
    "You are a parking-enforcement detection AI. "
    "Analyze camera frames strictly for two behaviours: "
    "(1) a person chalking a vehicle's tire, "
    "(2) a street sweeper vehicle. "
    "Respond with valid JSON only — no markdown, no commentary."
)

_FALLBACK = {
    "chalking_detected": False,
    "sweeper_detected": False,
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
        else:
            self._claude = None  # type: ignore[assignment]
            logger.info("VLM backend: Ollama (%s @ %s)", ollama_model, ollama_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes) -> dict:
        """Send a JPEG frame to the VLM and return the parsed JSON result."""
        try:
            if self._backend == "claude":
                return self._analyze_claude(image_bytes)
            return self._analyze_ollama(image_bytes)
        except Exception:
            logger.exception("VLM analysis error")
            return _FALLBACK.copy()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _analyze_claude(self, image_bytes: bytes) -> dict:
        b64 = base64.standard_b64encode(image_bytes).decode()

        response = self._claude.messages.create(
            model=self._claude_model,
            max_tokens=256,
            # Cache the static system prompt — saves ~75 % of input tokens on
            # repeated calls (cache TTL is 5 min, refreshed each hit).
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                }
            ],
        )
        return _parse_json(response.content[0].text)

    def _analyze_ollama(self, image_bytes: bytes) -> dict:
        b64 = base64.standard_b64encode(image_bytes).decode()

        payload = {
            "model": self._ollama_model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{_USER_PROMPT}",
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        resp = httpx.post(
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return _parse_json(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object in an LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        data = json.loads(text)
        return {
            "chalking_detected": bool(data.get("chalking_detected", False)),
            "sweeper_detected": bool(data.get("sweeper_detected", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "description": str(data.get("description", "")),
        }
    except json.JSONDecodeError:
        logger.warning("Could not parse VLM JSON: %r", text[:200])
        return _FALLBACK.copy()
