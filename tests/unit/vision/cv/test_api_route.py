"""Smoke test for POST /api/cv/takeoff."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from vision.cv import ocr_engines
from vision.cv.models import TextElement


class FakeEngine:
    def read(self, image, confidence_threshold):
        return [TextElement("KITCHEN", (270, 390, 380, 410), 0.95)]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(ocr_engines, "get_engine", lambda *_: FakeEngine())
    return TestClient(app)


def plan_png() -> bytes:
    img = np.full((800, 1000), 255, np.uint8)
    for a, b in [((150, 150), (850, 150)), ((850, 150), (850, 650)),
                 ((850, 650), (150, 650)), ((150, 650), (150, 150))]:
        cv2.line(img, a, b, 0, 8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_takeoff_route(client):
    response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"include_annotations": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["takeoff"]["schema_version"] == "1.0.0"
    assert body["takeoff"]["metadata"]["wall_count"] >= 4
    assert len(body["takeoff"]["rooms"]) == 1
    assert body["annotations"] is not None
    assert any(e["type"] == "wall" for e in body["annotations"]["elements"])


def test_takeoff_route_unsupported_mime(client):
    response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.zip", b"1234", "application/zip")},
    )
    assert response.status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
