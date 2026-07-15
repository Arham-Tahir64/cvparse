"""Smoke test for POST /api/cv/takeoff."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_store import get_model_repository
from vision.cv import ocr_engines
from vision.cv.models import TextElement
from vision.domain.repository import InMemoryModelRepository


class FakeEngine:
    def read(self, image, confidence_threshold):
        return [TextElement("KITCHEN", (270, 390, 380, 410), 0.95)]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(ocr_engines, "get_engine", lambda *_: FakeEngine())
    repository = InMemoryModelRepository()
    app.dependency_overrides[get_model_repository] = lambda: repository
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


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
    assert "model" not in body


def test_takeoff_route_can_include_editable_model(client):
    response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"include_model": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["takeoff"]["schema_version"] == "1.0.0"
    assert body["model"]["schema_version"] == "2.0.0-alpha.1"
    assert len(body["model"]["source"]["fingerprint"]) == 64
    assert body["model"]["validation_issues"][0]["code"] == "scale.unconfirmed"
    assert body["model"]["walls"]


def test_persisted_model_scale_review_and_revision_workflow(client):
    created_response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    )
    assert created_response.status_code == 200
    created = created_response.json()["model"]
    model_id = created["id"]
    wall_id = created["walls"][0]["id"]
    assert created["revision"] == 1

    fetched = client.get(f"/api/takeoff/models/{model_id}")
    assert fetched.status_code == 200
    assert fetched.json()["model"] == created

    scaled_response = client.put(
        f"/api/takeoff/models/{model_id}/scale",
        json={
            "expected_revision": 1,
            "pixels_per_unit": 20,
            "unit": "ft",
            "actor": "reviewer@example.test",
        },
    )
    assert scaled_response.status_code == 200
    scaled = scaled_response.json()["model"]
    assert scaled["revision"] == 2
    assert scaled["scale"]["review_status"] == "confirmed"
    assert all(wall["length"] is not None for wall in scaled["walls"])
    assert not any(
        issue["code"] == "scale.unconfirmed"
        for issue in scaled["validation_issues"]
    )
    assert scaled["edit_history"][-1]["action"] == "set_scale"

    reviewed_response = client.put(
        f"/api/takeoff/models/{model_id}/objects/{wall_id}/review",
        json={"expected_revision": 2, "status": "confirmed"},
    )
    assert reviewed_response.status_code == 200
    reviewed = reviewed_response.json()["model"]
    reviewed_wall = next(wall for wall in reviewed["walls"] if wall["id"] == wall_id)
    assert reviewed["revision"] == 3
    assert reviewed_wall["metadata"]["review_status"] == "confirmed"
    assert reviewed_wall["metadata"]["locked"] is True
    assert reviewed["edit_history"][-1]["action"] == "set_review_status"

    stale = client.put(
        f"/api/takeoff/models/{model_id}/scale",
        json={"expected_revision": 2, "pixels_per_unit": 21, "unit": "ft"},
    )
    assert stale.status_code == 409

    duplicate_create = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    )
    assert duplicate_create.status_code == 409


def test_persisted_wall_endpoint_edit_updates_shared_geometry(client):
    created_response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    )
    created = created_response.json()["model"]
    shared = next(node for node in created["nodes"] if len(node["connected_wall_ids"]) > 1)
    wall = next(
        wall for wall in created["walls"]
        if wall["start_node_id"] == shared["id"] or wall["end_node_id"] == shared["id"]
    )
    endpoint = "start" if wall["start_node_id"] == shared["id"] else "end"
    response = client.put(
        f"/api/takeoff/models/{created['id']}/walls/{wall['id']}/endpoints/{endpoint}",
        json={
            "expected_revision": 1,
            "x": shared["point"]["x"] + 2,
            "y": shared["point"]["y"] + 2,
            "actor": "reviewer@example.test",
        },
    )

    assert response.status_code == 200
    updated = response.json()["model"]
    assert updated["revision"] == 2
    moved = next(node for node in updated["nodes"] if node["id"] == shared["id"])
    assert moved["point"] == {
        "x": shared["point"]["x"] + 2,
        "y": shared["point"]["y"] + 2,
    }
    assert updated["edit_history"][-1]["action"] == "move_wall_endpoint"

    annotations_response = client.get(
        f"/api/takeoff/models/{created['id']}/annotations"
    )
    assert annotations_response.status_code == 200
    annotations = annotations_response.json()["annotations"]
    assert annotations["model_revision"] == 2
    rendered_wall = next(
        item for item in annotations["elements"] if item["id"] == wall["id"]
    )
    coordinate_suffix = "1" if endpoint == "start" else "2"
    centerline = rendered_wall["geometry"]["centerline"]
    assert centerline[f"x{coordinate_suffix}"] == shared["point"]["x"] + 2
    assert centerline[f"y{coordinate_suffix}"] == shared["point"]["y"] + 2
    assert rendered_wall["review_state"] == "needs_review"

    svg_response = client.get(
        f"/api/takeoff/models/{created['id']}/overlay.svg"
    )
    assert svg_response.status_code == 200
    assert svg_response.headers["content-type"].startswith("image/svg+xml")
    assert svg_response.headers["etag"] == f'"{created["id"]}:2"'
    assert f'data-id="{wall["id"]}"' in svg_response.text

    quantities_response = client.get(
        f"/api/takeoff/models/{created['id']}/quantities?basis=provisional"
    )
    assert quantities_response.status_code == 200
    quantities = quantities_response.json()["quantities"]
    assert quantities["model_revision"] == 2
    assert quantities["basis"] == "provisional"
    assert quantities["authoritative"] is False
    assert quantities["counts"]["walls"] == len(updated["walls"])
    assert quantities["pixel_measurements"]["wall_centerline_length_px"] == pytest.approx(
        sum(item["length_px"] for item in updated["walls"])
    )
    assert quantities["pixel_measurements"]["wall_centerline_length_px"] != pytest.approx(
        sum(item["length_px"] for item in created["walls"])
    )


def test_takeoff_route_unsupported_mime(client):
    response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.zip", b"1234", "application/zip")},
    )
    assert response.status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
