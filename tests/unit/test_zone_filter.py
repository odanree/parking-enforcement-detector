import pytest
import numpy as np
from src.detection.zone_filter import ZoneFilter
from src.detection.object_detector import Detection


def make_detection(track_id: int, cx: int, cy: int) -> Detection:
    half = 30
    return Detection(
        track_id=track_id,
        class_name="person",
        confidence=0.90,
        bbox=(cx - half, cy - half, cx + half, cy + half),
    )


SQUARE_ZONE = [[100, 100], [500, 100], [500, 500], [100, 500]]


class TestZoneFilter:
    def setup_method(self):
        self.zf = ZoneFilter(zones={"street_zone": SQUARE_ZONE})

    def test_inside_point_passes(self):
        det = make_detection(1, 300, 300)    # center of square
        result = self.zf.filter([det], "street_zone")
        assert result == [det]

    def test_outside_point_rejected(self):
        det = make_detection(2, 50, 50)      # top-left, outside square
        result = self.zf.filter([det], "street_zone")
        assert result == []

    def test_edge_point_passes(self):
        det = make_detection(3, 100, 300)    # on left edge
        result = self.zf.filter([det], "street_zone")
        assert len(result) == 1

    def test_unknown_zone_returns_all(self):
        dets = [make_detection(i, 300, 300) for i in range(3)]
        result = self.zf.filter(dets, "nonexistent_zone")
        assert result == dets

    def test_multiple_detections_mixed(self):
        inside = make_detection(10, 300, 300)
        outside = make_detection(11, 600, 600)
        result = self.zf.filter([inside, outside], "street_zone")
        assert result == [inside]
