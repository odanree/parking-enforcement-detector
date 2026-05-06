"""Unit tests for AppState — the shared state between the pipeline and web layer."""

import time
import pytest
from src.web.state import AppState


class TestRecordAlert:
    def setup_method(self):
        self.state = AppState()

    def test_chalking_increments_counter(self):
        self.state.record_alert("chalking", 0.9, "test")
        stats = self.state.get_stats()
        assert stats["total_chalking"] == 1
        assert stats["total_sweeper"] == 0

    def test_sweeper_increments_counter(self):
        self.state.record_alert("sweeper", 0.8, "sweeper test")
        stats = self.state.get_stats()
        assert stats["total_sweeper"] == 1

    def test_event_appears_in_get_events(self):
        self.state.record_alert("chalking", 0.75, "chalk mark on tire")
        events = self.state.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "chalking"
        assert events[0]["confidence"] == pytest.approx(0.75)
        assert events[0]["description"] == "chalk mark on tire"

    def test_most_recent_event_first(self):
        self.state.record_alert("chalking", 0.7, "first")
        self.state.record_alert("sweeper", 0.8, "second")
        events = self.state.get_events()
        assert events[0]["event_type"] == "sweeper"
        assert events[1]["event_type"] == "chalking"

    def test_snapshot_url_formatted(self):
        self.state.record_alert("chalking", 0.9, "desc", snapshot="chalking_001.jpg")
        events = self.state.get_events()
        assert events[0]["snapshot_url"] == "/snapshots/chalking_001.jpg"

    def test_no_snapshot_gives_none_url(self):
        self.state.record_alert("chalking", 0.9, "desc")
        events = self.state.get_events()
        assert events[0]["snapshot_url"] is None


class TestPendingVlm:
    def setup_method(self):
        self.state = AppState()

    def test_add_and_retrieve_job(self):
        self.state.add_pending_vlm("chalking", 42, "b64thumb")
        jobs = self.state.get_pending_vlm()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "chalking_42"
        assert jobs[0]["kind"] == "chalking"

    def test_complete_marks_job(self):
        self.state.add_pending_vlm("chalking", 7, "thumb")
        self.state.complete_pending_vlm("chalking", 7, detected=True)
        jobs = self.state.get_pending_vlm()
        assert jobs[0]["detected"] is True
        assert "completed_at" in jobs[0]

    def test_completed_jobs_evicted_after_4s(self):
        self.state.add_pending_vlm("chalking", 99, "thumb")
        self.state.complete_pending_vlm("chalking", 99, detected=False)
        # Manually backdate completed_at so the cleanup fires
        with self.state._lock:
            self.state._pending_vlm[0]["completed_at"] -= 5.0
        jobs = self.state.get_pending_vlm()
        assert len(jobs) == 0

    def test_duplicate_job_id_replaces_entry(self):
        self.state.add_pending_vlm("chalking", 1, "thumb_a")
        self.state.add_pending_vlm("chalking", 1, "thumb_b")
        jobs = self.state.get_pending_vlm()
        assert len(jobs) == 1
        assert jobs[0]["thumbnail"] == "thumb_b"

    def test_sample_count_increments(self):
        self.state.add_pending_vlm("chalking", 5, "t")
        self.state.add_pending_vlm("chalking", 5, "t")
        jobs = self.state.get_pending_vlm()
        assert jobs[0]["sample_num"] == 2


class TestIsLive:
    def test_no_stream_is_live(self):
        state = AppState()
        assert state.get_stats()["is_live"] is True

    def test_stream_with_speed_attr_not_live(self):
        state = AppState()

        class FakeFileStream:
            _speed = 1.0
            _direction = 1

        state._stream = FakeFileStream()
        assert state.get_stats()["is_live"] is False

    def test_stream_without_speed_attr_is_live(self):
        state = AppState()

        class FakeRtspStream:
            pass

        state._stream = FakeRtspStream()
        assert state.get_stats()["is_live"] is True


class TestTogglePause:
    def test_pause_toggles(self):
        state = AppState()
        assert state.paused is False
        state.toggle_pause()
        assert state.paused is True
        state.toggle_pause()
        assert state.paused is False
