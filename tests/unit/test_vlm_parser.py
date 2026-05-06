"""Unit tests for the VLM JSON parser (_parse_json)."""

import pytest
from src.vlm.analyzer import _parse_json


VALID_JSON = (
    '{"chalking_detected": true, "sweeper_detected": false, '
    '"pe_vehicle_detected": false, "confidence": 0.87, '
    '"description": "Person crouching near rear tire"}'
)


class TestParseJson:
    def test_plain_json(self):
        result = _parse_json(VALID_JSON)
        assert result["chalking_detected"] is True
        assert result["confidence"] == pytest.approx(0.87)
        assert result["description"] == "Person crouching near rear tire"

    def test_strips_json_fence(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        result = _parse_json(fenced)
        assert result["chalking_detected"] is True

    def test_strips_plain_fence(self):
        fenced = f"```\n{VALID_JSON}\n```"
        result = _parse_json(fenced)
        assert result["chalking_detected"] is True

    def test_invalid_json_returns_fallback(self):
        result = _parse_json("this is not json at all")
        assert result["chalking_detected"] is False
        assert result["confidence"] == 0.0
        assert result["description"] == "VLM analysis failed"

    def test_empty_string_returns_fallback(self):
        result = _parse_json("")
        assert result["chalking_detected"] is False

    def test_missing_fields_get_defaults(self):
        result = _parse_json('{"chalking_detected": true}')
        assert result["sweeper_detected"] is False
        assert result["pe_vehicle_detected"] is False
        assert result["confidence"] == 0.0
        assert result["description"] == ""

    def test_confidence_coerced_to_float(self):
        result = _parse_json('{"confidence": "0.9", "chalking_detected": false, '
                             '"sweeper_detected": false, "pe_vehicle_detected": false, '
                             '"description": "test"}')
        assert isinstance(result["confidence"], float)

    def test_chalking_coerced_to_bool(self):
        result = _parse_json('{"chalking_detected": 1, "sweeper_detected": 0, '
                             '"pe_vehicle_detected": 0, "confidence": 0.5, "description": ""}')
        assert result["chalking_detected"] is True
        assert result["sweeper_detected"] is False

    def test_all_fields_present(self):
        result = _parse_json(VALID_JSON)
        assert set(result.keys()) == {
            "chalking_detected",
            "sweeper_detected",
            "pe_vehicle_detected",
            "confidence",
            "description",
        }
