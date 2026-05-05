import time
import pytest
from src.behavior.chalking_analyzer import ChalkingAnalyzer


def feed(analyzer, track_id, heights):
    results = []
    for h in heights:
        results.append(analyzer.update(track_id, h))
    return results


class TestChalkingAnalyzer:
    def test_no_trigger_on_insufficient_history(self):
        a = ChalkingAnalyzer(height_decrease_threshold=0.30, history_frames=15)
        # Only 2 frames — not enough history
        assert not a.update(1, 200)
        assert not a.update(1, 150)

    def test_triggers_on_30pct_decrease(self):
        a = ChalkingAnalyzer(height_decrease_threshold=0.30, history_frames=9, cooldown_seconds=0)
        # Establish a baseline of ~200 over 3 frames, then drop to 130 (35 % decrease)
        heights = [200, 200, 200, 200, 200, 200, 200, 200, 130]
        results = feed(a, 1, heights)
        assert results[-1] is True

    def test_no_trigger_on_small_decrease(self):
        a = ChalkingAnalyzer(height_decrease_threshold=0.30, history_frames=9, cooldown_seconds=0)
        heights = [200, 200, 200, 200, 200, 200, 200, 200, 180]  # 10 % decrease
        results = feed(a, 1, heights)
        assert not any(results)

    def test_cooldown_prevents_duplicate_alerts(self):
        a = ChalkingAnalyzer(height_decrease_threshold=0.30, history_frames=9, cooldown_seconds=999)
        heights = [200] * 6 + [130, 130, 130]
        results = feed(a, 1, heights)
        first_trigger = results.index(True)
        # All subsequent frames should be suppressed by cooldown
        assert not any(results[first_trigger + 1:])

    def test_evict_clears_history(self):
        a = ChalkingAnalyzer(history_frames=9, cooldown_seconds=0)
        feed(a, 42, [200, 200, 200])
        a.evict(42)
        assert 42 not in a._heights

    def test_multiple_tracks_independent(self):
        a = ChalkingAnalyzer(height_decrease_threshold=0.30, history_frames=9, cooldown_seconds=0)
        heights_trigger = [200] * 6 + [130, 130, 130]
        heights_stable = [200] * 9
        r1 = feed(a, 1, heights_trigger)
        r2 = feed(a, 2, heights_stable)
        assert any(r1)
        assert not any(r2)
