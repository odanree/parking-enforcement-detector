from datetime import datetime
import pytest
from src.behavior.sweeper_analyzer import SweeperAnalyzer

SCHEDULE = [{"weekday": 1, "start_time": "08:00", "end_time": "13:00"}]  # Tuesday


class TestSweeperAnalyzer:
    def setup_method(self):
        self.sa = SweeperAnalyzer(
            schedule=SCHEDULE,
            min_velocity=3.0,
            max_velocity=20.0,
            sustained_frames=5,
        )

    def test_sweep_window_true_on_tuesday_morning(self):
        tuesday_9am = datetime(2025, 6, 3, 9, 0)   # June 3 2025 = Tuesday
        assert self.sa.is_sweep_window(tuesday_9am) is True

    def test_sweep_window_false_on_tuesday_evening(self):
        tuesday_3pm = datetime(2025, 6, 3, 15, 0)
        assert self.sa.is_sweep_window(tuesday_3pm) is False

    def test_sweep_window_false_on_monday(self):
        monday = datetime(2025, 6, 2, 10, 0)
        assert self.sa.is_sweep_window(monday) is False

    def test_triggers_after_sustained_velocity(self):
        # Move 10 px/frame for 6 frames (> sustained_frames=5)
        results = []
        x = 0
        for _ in range(6):
            x += 10
            results.append(self.sa.update(1, (x, 300)))
        assert results[-1] is True

    def test_no_trigger_on_too_fast(self):
        results = []
        x = 0
        for _ in range(10):
            x += 50   # 50 px/frame — above max_velocity
            results.append(self.sa.update(1, (x, 300)))
        assert not any(results)

    def test_no_trigger_on_too_slow(self):
        results = []
        x = 0
        for _ in range(10):
            x += 1    # 1 px/frame — below min_velocity
            results.append(self.sa.update(1, (x, 300)))
        assert not any(results)

    def test_resets_count_on_velocity_out_of_range(self):
        # 4 good frames, then one fast frame, then 4 good frames → should not trigger
        sa = SweeperAnalyzer(SCHEDULE, min_velocity=3.0, max_velocity=20.0, sustained_frames=5)
        centers = [(i * 10, 300) for i in range(4)]   # good
        centers.append((centers[-1][0] + 100, 300))    # too fast
        centers += [(centers[-1][0] + 10 * i, 300) for i in range(1, 5)]  # good again
        results = [sa.update(1, c) for c in centers]
        assert not any(results)

    def test_evict_clears_state(self):
        self.sa.update(5, (0, 0))
        self.sa.evict(5)
        assert 5 not in self.sa._centers
        assert 5 not in self.sa._in_range_count
