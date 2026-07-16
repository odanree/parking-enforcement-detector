"""Unit tests for DetectionStrategy factory + strategy invariants."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.detection.strategy import (
    ParkingEnforcementStrategy,
    PositiveEvent,
    RodentStrategy,
    strategy_from_env,
)


class TestStrategyFromEnv:
    def test_default_is_parking(self, monkeypatch):
        monkeypatch.delenv("DETECTION_MODE", raising=False)
        s = strategy_from_env()
        assert s.name == "parking"
        assert s.use_parking_gates is True
        assert s.alert_category == "chalking"

    def test_rodent_mode(self, monkeypatch):
        monkeypatch.setenv("DETECTION_MODE", "rodent")
        s = strategy_from_env()
        assert s.name == "rodent"
        assert s.use_parking_gates is False
        assert s.alert_category == "rodent"

    def test_unknown_mode_falls_back_to_parking(self, monkeypatch):
        monkeypatch.setenv("DETECTION_MODE", "wombat")
        s = strategy_from_env()
        assert s.name == "parking"


class TestParkingStrategy:
    def test_on_positive_is_null_object(self):
        s = ParkingEnforcementStrategy()
        # Should never raise, never touch external state.
        s.on_positive(PositiveEvent(
            track_id=1, bbox=(0, 0, 10, 10), frame_width=1280, frame_height=720, confidence=0.9
        ))

    def test_uses_default_vlm_prompts(self):
        # None → VLMAnalyzer keeps its module-level (chalking) defaults.
        assert ParkingEnforcementStrategy().vlm_prompts() == (None, None)


class TestRodentStrategy:
    def test_rodent_prompts_are_present(self):
        user, system = RodentStrategy().vlm_prompts()
        assert user is not None and "rodent" in user.lower()
        assert system is not None and "rodent" in system.lower()

    def test_on_positive_noop_when_slew_disabled(self, monkeypatch):
        monkeypatch.setenv("RODENT_SLEW_ENABLED", "false")
        s = RodentStrategy()
        # Would normally import slew — must not, when disabled.
        with patch("src.stream.slew.get_dispatcher") as mock_get:
            s.on_positive(PositiveEvent(
                track_id=1, bbox=(0, 0, 10, 10), frame_width=1280, frame_height=720, confidence=0.9
            ))
            mock_get.assert_not_called()

    def test_on_positive_calls_slew_when_enabled(self, monkeypatch):
        monkeypatch.setenv("RODENT_SLEW_ENABLED", "true")
        s = RodentStrategy()
        with patch("src.stream.slew.get_dispatcher") as mock_get:
            mock_disp = mock_get.return_value
            s.on_positive(PositiveEvent(
                track_id=42, bbox=(100, 100, 200, 200),
                frame_width=1280, frame_height=720, confidence=0.9,
            ))
            mock_get.assert_called_once()
            mock_disp.slew_to_bbox.assert_called_once()
            args, kwargs = mock_disp.slew_to_bbox.call_args
            assert kwargs["bbox"] == (100, 100, 200, 200)
            assert kwargs["event_key"] == ("rodent", 42)

    def test_on_positive_swallows_slew_exception(self, monkeypatch):
        # A slew failure must not crash the pipeline harvest loop.
        monkeypatch.setenv("RODENT_SLEW_ENABLED", "true")
        s = RodentStrategy()
        with patch("src.stream.slew.get_dispatcher", side_effect=RuntimeError("boom")):
            s.on_positive(PositiveEvent(
                track_id=1, bbox=(0, 0, 10, 10),
                frame_width=1280, frame_height=720, confidence=0.9,
            ))  # must not raise
