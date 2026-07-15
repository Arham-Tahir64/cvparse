"""Smoke test for POST /api/cv/takeoff."""
import cv2
import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_store import get_model_repository, get_source_asset_repository
from vision.cv import ocr_engines
from vision.cv.models import TextElement
from vision.domain.repository import InMemoryModelRepository
from vision.domain.source_assets import InMemorySourceAssetRepository


class FakeEngine:
    def read(self, image, confidence_threshold):
        return [TextElement("KITCHEN", (270, 390, 380, 410), 0.95)]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(ocr_engines, "get_engine", lambda *_: FakeEngine())
    repository = InMemoryModelRepository()
    source_repository = InMemorySourceAssetRepository()
    app.dependency_overrides[get_model_repository] = lambda: repository
    app.dependency_overrides[get_source_asset_repository] = lambda: source_repository
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
    assert body["model"]["schema_version"] == "2.0.0-alpha.2"
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

    incomplete_approval = client.put(
        f"/api/takeoff/models/{model_id}/approval",
        json={"expected_revision": 3, "status": "approved"},
    )
    assert incomplete_approval.status_code == 422
    assert "cannot be approved" in incomplete_approval.json()["detail"]


def test_persisted_model_undo_redo_history_and_branching(client):
    created = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    ).json()["model"]
    model_id = created["id"]
    wall_id = created["walls"][0]["id"]
    scaled = client.put(
        f"/api/takeoff/models/{model_id}/scale",
        json={"expected_revision": 1, "pixels_per_unit": 20, "unit": "ft"},
    ).json()["model"]
    reviewed = client.put(
        f"/api/takeoff/models/{model_id}/objects/{wall_id}/review",
        json={"expected_revision": 2, "status": "confirmed"},
    ).json()["model"]

    stale = client.post(
        f"/api/takeoff/models/{model_id}/undo",
        json={"expected_revision": 2},
    )
    assert stale.status_code == 409
    undone_response = client.post(
        f"/api/takeoff/models/{model_id}/undo",
        json={"expected_revision": 3, "actor": "undo-user"},
    )
    assert undone_response.status_code == 200
    undone = undone_response.json()["model"]
    undone_wall = next(wall for wall in undone["walls"] if wall["id"] == wall_id)
    assert undone["revision"] == 4
    assert undone_wall["metadata"]["review_status"] == "likely_correct"
    assert undone["undo_revision_stack"] == [1]
    assert undone["redo_revision_stack"] == [3]
    assert undone["edit_history"][-1]["action"] == "undo"

    historical_response = client.get(
        f"/api/takeoff/models/{model_id}/revisions/3"
    )
    assert historical_response.status_code == 200
    historical = historical_response.json()["model"]
    historical_wall = next(
        wall for wall in historical["walls"] if wall["id"] == wall_id
    )
    assert historical_wall["metadata"]["review_status"] == "confirmed"
    assert client.get(
        f"/api/takeoff/models/{model_id}/revisions/999"
    ).status_code == 409

    redone_response = client.post(
        f"/api/takeoff/models/{model_id}/redo",
        json={"expected_revision": 4, "actor": "redo-user"},
    )
    assert redone_response.status_code == 200
    redone = redone_response.json()["model"]
    redone_wall = next(wall for wall in redone["walls"] if wall["id"] == wall_id)
    assert redone["revision"] == 5
    assert redone_wall["metadata"]["review_status"] == "confirmed"
    assert redone["redo_revision_stack"] == []
    assert redone["edit_history"][-1]["action"] == "redo"

    undone_again = client.post(
        f"/api/takeoff/models/{model_id}/undo",
        json={"expected_revision": 5},
    ).json()["model"]
    branched = client.put(
        f"/api/takeoff/models/{model_id}/scale",
        json={"expected_revision": 6, "pixels_per_unit": 21, "unit": "ft"},
    ).json()["model"]
    assert undone_again["redo_revision_stack"]
    assert branched["revision"] == 7
    assert branched["redo_revision_stack"] == []
    no_redo = client.post(
        f"/api/takeoff/models/{model_id}/redo",
        json={"expected_revision": 7},
    )
    assert no_redo.status_code == 422
    assert no_redo.json()["detail"] == "nothing to redo"


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

    reviewed_pdf_response = client.get(
        f"/api/takeoff/models/{created['id']}/reviewed.pdf"
    )
    assert reviewed_pdf_response.status_code == 200
    assert reviewed_pdf_response.headers["content-type"] == "application/pdf"
    assert reviewed_pdf_response.headers["etag"] == f'"{created["id"]}:2:pdf"'
    reviewed_pdf = fitz.open(
        stream=reviewed_pdf_response.content, filetype="pdf",
    )
    assert reviewed_pdf.page_count == 1
    assert reviewed_pdf.metadata["subject"] == f"Model {created['id']} revision 2"
    assert reviewed_pdf[0].get_drawings()
    reviewed_pdf.close()


def test_persisted_wall_create_delete_quantities_and_undo(client):
    created = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    ).json()["model"]
    model_id = created["id"]
    initial_length = sum(wall["length_px"] for wall in created["walls"])

    create_response = client.post(
        f"/api/takeoff/models/{model_id}/walls",
        json={
            "expected_revision": 1,
            "start_x": 10,
            "start_y": 10,
            "end_x": 100,
            "end_y": 10,
            "thickness_px": 8,
            "wall_type": "interior",
            "snap_tolerance_px": 0,
            "actor": "wall-editor",
        },
    )
    assert create_response.status_code == 200
    added = create_response.json()["model"]
    wall_id = added["edit_history"][-1]["payload"]["wall_id"]
    wall = next(item for item in added["walls"] if item["id"] == wall_id)
    assert added["revision"] == 2
    assert wall["metadata"]["source"]["kind"] == "manual_created"
    assert wall["length_px"] == 90
    quantities = client.get(
        f"/api/takeoff/models/{model_id}/quantities"
    ).json()["quantities"]
    assert quantities["model_revision"] == 2
    assert quantities["pixel_measurements"]["wall_centerline_length_px"] == pytest.approx(
        initial_length + 90
    )

    delete_response = client.request(
        "DELETE",
        f"/api/takeoff/models/{model_id}/walls/{wall_id}",
        json={"expected_revision": 2, "actor": "wall-editor"},
    )
    assert delete_response.status_code == 200
    deleted = delete_response.json()["model"]
    assert deleted["revision"] == 3
    assert not any(item["id"] == wall_id for item in deleted["walls"])
    assert deleted["edit_history"][-1]["action"] == "delete_wall"

    restored_response = client.post(
        f"/api/takeoff/models/{model_id}/undo",
        json={"expected_revision": 3, "actor": "wall-editor"},
    )
    assert restored_response.status_code == 200
    restored = restored_response.json()["model"]
    assert restored["revision"] == 4
    assert any(item["id"] == wall_id for item in restored["walls"])


def test_persisted_wall_split_preserves_length_and_is_undoable(client):
    created = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.png", plan_png(), "image/png")},
        data={"persist_model": "true"},
    ).json()["model"]
    model_id = created["id"]
    added = client.post(
        f"/api/takeoff/models/{model_id}/walls",
        json={
            "expected_revision": 1,
            "start_x": 10,
            "start_y": 10,
            "end_x": 100,
            "end_y": 10,
            "thickness_px": 8,
            "snap_tolerance_px": 0,
        },
    ).json()["model"]
    wall_id = added["edit_history"][-1]["payload"]["wall_id"]

    split_response = client.post(
        f"/api/takeoff/models/{model_id}/walls/{wall_id}/split",
        json={
            "expected_revision": 2,
            "x": 55,
            "y": 13,
            "projection_tolerance_px": 4,
            "actor": "wall-editor",
        },
    )
    assert split_response.status_code == 200
    split = split_response.json()["model"]
    event = split["edit_history"][-1]
    child_id = event["payload"]["new_wall_id"]
    parent = next(item for item in split["walls"] if item["id"] == wall_id)
    child = next(item for item in split["walls"] if item["id"] == child_id)
    assert split["revision"] == 3
    assert parent["length_px"] == 45
    assert child["length_px"] == 45
    assert parent["metadata"]["source"]["kind"] == "manual_adjusted"
    assert child["metadata"]["source"]["kind"] == "manual_created"
    quantities = client.get(
        f"/api/takeoff/models/{model_id}/quantities"
    ).json()["quantities"]
    split_pair_length = sum(
        item["length_px"] for item in split["walls"]
        if item["id"] in {wall_id, child_id}
    )
    assert split_pair_length == 90
    assert quantities["model_revision"] == 3

    undo_response = client.post(
        f"/api/takeoff/models/{model_id}/undo",
        json={"expected_revision": 3},
    )
    assert undo_response.status_code == 200
    undone = undo_response.json()["model"]
    restored = next(item for item in undone["walls"] if item["id"] == wall_id)
    assert restored["length_px"] == 90
    assert not any(item["id"] == child_id for item in undone["walls"])


def test_takeoff_route_unsupported_mime(client):
    response = client.post(
        "/api/cv/takeoff",
        files={"file": ("plan.zip", b"1234", "application/zip")},
    )
    assert response.status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
