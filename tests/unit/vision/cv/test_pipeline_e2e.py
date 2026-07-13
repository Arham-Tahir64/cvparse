"""End-to-end pipeline run on a synthetic two-room plan with a door arc."""
import cv2
import numpy as np
import pytest

from vision.cv import ocr_engines, pipeline
from vision.cv.config import PipelineConfig
from vision.cv.models import TextElement


class FakeEngine:
    name = "fake"

    def __init__(self, elements):
        self.elements = elements

    def read(self, image, confidence_threshold):
        return [e for e in self.elements if e.confidence >= confidence_threshold]


@pytest.fixture()
def synthetic_plan():
    """White sheet, two-room plan with double-line walls and a door arc."""
    img = np.full((800, 1000), 255, np.uint8)

    def dwall(x1, y1, x2, y2, t=8):
        cv2.line(img, (x1, y1), (x2, y2), 0, t)

    # outer walls (thick strokes read as wall bodies)
    dwall(150, 150, 850, 150)
    dwall(850, 150, 850, 650)
    dwall(850, 650, 150, 650)
    dwall(150, 650, 150, 150)
    # continuous divider (open passages would merge the rooms into one face)
    dwall(500, 150, 500, 650)
    # door arc against the divider, hinge mid-wall
    cv2.ellipse(img, (500, 460), (60, 60), 0, 180, 270, 0, 2)
    return img


def test_full_pipeline(synthetic_plan, monkeypatch):
    texts = [
        TextElement("KITCHEN", (270, 390, 380, 410), 0.95),
        TextElement("LIVING", (630, 390, 720, 410), 0.93),
    ]
    monkeypatch.setattr(ocr_engines, "get_engine", lambda *_: FakeEngine(texts))

    result = pipeline.run_pipeline(
        image=synthetic_plan,
        config=PipelineConfig(hough_circles_param2=25.0),
    )

    assert len(result.walls) >= 4
    for wall in result.walls:
        assert wall.merge_kind in ("paired_faces", "single_face")
        assert wall.source_ids
        assert 0 < wall.visual_thickness <= 45
    assert len(result.doors) == 1
    assert abs(result.doors[0].position.x - 500) < 6
    assert abs(result.doors[0].position.y - 460) < 6
    assert any(g.kind == "door" for g in result.gaps)
    assert len(result.rooms) >= 2
    labels = {r.label for r in result.rooms if r.label}
    assert "KITCHEN" in labels
    assert "LIVING" in labels
    assert all(t >= 0 for t in result.debug.stage_timings.values())
    assert len(result.debug.stage_timings) == 10

    # serialization round trip
    import json

    from vision.cv import serialize
    data = serialize.to_json_dict(result)
    json.dumps(data)
    assert data["schema_version"] == "1.0.0"
