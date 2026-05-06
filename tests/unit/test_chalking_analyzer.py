import time
import pytest
from src.behavior.chalking_analyzer import ChalkingAnalyzer


class TestChalkingAnalyzer:
    def test_no_trigger_before_entry_frames(self):
        a = ChalkingAnalyzer(entry_frames=10, sample_every_n=30, cooldown_seconds=0)
        for _ in range(9):
            assert not a.update(1)

    def test_triggers_at_entry_frame(self):
        a = ChalkingAnalyzer(entry_frames=5, sample_every_n=30, cooldown_seconds=0)
        results = [a.update(1) for _ in range(5)]
        assert results[-1] is True

    def test_triggers_every_sample_n_frames_after_entry(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=0)
        # Run 20 frames, collect indices where True is returned
        trigger_indices = [i for i in range(20) if a.update(1)]
        # Should fire at frame 2 (entry), then 7, 12, 17 (every 5 after)
        assert trigger_indices == [2, 7, 12, 17]

    def test_cooldown_suppresses_after_alert(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=999)
        # Get first trigger
        for _ in range(3):
            a.update(1)
        a.on_alert(1)
        # Next sampling points should all be suppressed
        for _ in range(20):
            assert not a.update(1)

    def test_cooldown_zero_allows_immediate_retrigger(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=0)
        # Advance to first trigger
        for _ in range(3):
            a.update(1)
        a.on_alert(1)
        # advance to next sampling point
        for _ in range(4):
            a.update(1)
        assert a.update(1) is True

    def test_evict_clears_state(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=0)
        for _ in range(5):
            a.update(42)
        a.evict(42)
        assert 42 not in a._frame_count
        assert 42 not in a._last_alert

    def test_multiple_tracks_independent(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=0)
        # Track 1: advance to trigger
        r1 = [a.update(1) for _ in range(3)]
        # Track 2: only 2 frames — should not trigger
        r2 = [a.update(2) for _ in range(2)]
        assert r1[-1] is True
        assert not any(r2)

    def test_cooldown_per_track(self):
        a = ChalkingAnalyzer(entry_frames=3, sample_every_n=5, cooldown_seconds=999)
        for _ in range(3):
            a.update(1)
        a.on_alert(1)
        # Track 2 is independent — no cooldown applied
        r2 = [a.update(2) for _ in range(3)]
        assert r2[-1] is True
