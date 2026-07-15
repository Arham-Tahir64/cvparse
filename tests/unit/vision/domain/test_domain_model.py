"""Phase 1 editable-domain import, validation, and serialization tests."""
from __future__ import annotations

import copy
import json

from vision.cv import serialize as legacy_serialize
from vision.cv.models import (
    CVTakeoffResult,
    DebugInfo,
    Door as CVDoor,
    Gap,
    LineSegment,
    PlanMetadata,
    Point,
    Room as CVRoom,
    Wall as CVWall,
    Window as CVWindow,
)
from vision.domain.import_cv import import_cv_result
from vision.domain.models import ObjectSourceKind, OpeningKind, ReviewStatus
from vision.domain.serialize import from_json_dict, to_json_dict
from vision.domain.validation import validate_model


def cv_result() -> CVTakeoffResult:
    first_line = LineSegment(Point(20, 20), Point(220, 20), thickness=12)
    second_line = LineSegment(Point(220, 20), Point(220, 220), thickness=12)
    walls = [
        CVWall(
            "W0001", "H", first_line, 12, visual_thickness=14,
            merge_kind="paired_faces", fit_support_ratio=0.91,
            merge_confidence=0.88, source_ids=["L1", "L2"],
            length_px=first_line.length,
        ),
        CVWall(
            "W0002", "V", second_line, 12, visual_thickness=14,
            merge_kind="paired_faces", fit_support_ratio=0.86,
            merge_confidence=0.82, source_ids=["L3", "L4"],
            length_px=second_line.length,
        ),
    ]
    gaps = [
        Gap(
            "G0001", "W0001", "H", Point(100, 20), 40,
            (80, 0, 120, 40), "door", 0.8, 0.1,
        ),
        Gap(
            "G0002", "W0002", "V", Point(220, 110), 30,
            (200, 95, 240, 125), "window", 0.85, 0.1,
        ),
    ]
    door = CVDoor(
        "D0001", Point(100, 20), Point(100, 60), 40, "W0001", "cw",
        swing_arc=[Point(140, 20), Point(128, 48), Point(100, 60)],
        confidence=0.84,
    )
    window = CVWindow("WD0001", Point(220, 110), 30, "W0002")
    room = CVRoom(
        "R0001",
        [Point(20, 20), Point(220, 20), Point(220, 220), Point(20, 220)],
        label="OFFICE", label_confidence=0.92, area=40000,
    )
    return CVTakeoffResult(
        walls=walls, gaps=gaps, doors=[door], windows=[window], rooms=[room],
        metadata=PlanMetadata("plan.pdf", 500, 400, 200, 0, 2, 1),
        debug=DebugInfo(),
    )


def test_import_builds_one_authoritative_opening_per_gap():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    assert len(model.openings) == 2
    assert {item.kind for item in model.openings} == {
        OpeningKind.DOOR, OpeningKind.WINDOW,
    }
    assert model.doors[0].opening_id in {item.id for item in model.openings}
    assert model.windows[0].opening_id in {item.id for item in model.openings}
    assert sum(len(wall.opening_ids) for wall in model.walls) == 2


def test_import_reconstructs_shared_graph_node_and_connectivity():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    shared = [node for node in model.nodes if len(node.connected_wall_ids) == 2]
    assert len(shared) == 1
    assert all(len(wall.connected_wall_ids) == 1 for wall in model.walls)
    assert all(wall.start_node_id != wall.end_node_id for wall in model.walls)


def test_import_ids_are_deterministic_for_same_source_and_geometry():
    first = import_cv_result(cv_result(), source_fingerprint="abc123")
    second = import_cv_result(cv_result(), source_fingerprint="abc123")

    assert first.id == second.id
    assert [wall.id for wall in first.walls] == [wall.id for wall in second.walls]
    assert [opening.id for opening in first.openings] == [
        opening.id for opening in second.openings
    ]


def test_automatic_import_is_reviewable_and_preserves_provenance():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    first_wall = model.walls[0]

    assert first_wall.metadata.source.kind == ObjectSourceKind.AUTOMATIC_DETECTED
    assert set(first_wall.metadata.source.detector_ids) == {"W0001", "L1", "L2"}
    assert first_wall.metadata.review_status == ReviewStatus.LIKELY_CORRECT
    assert all(
        item.metadata.review_status != ReviewStatus.CONFIRMED
        for collection in (
            model.nodes, model.walls, model.openings,
            model.doors, model.windows, model.rooms,
        )
        for item in collection
    )


def test_unconfirmed_scale_is_highest_priority_issue():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    assert model.validation_issues
    assert model.validation_issues[0].code == "scale.unconfirmed"
    assert model.validation_issues[0].priority == 1.0


def test_duplicate_opening_ranges_are_flagged_not_silently_merged():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    duplicate = copy.deepcopy(model.openings[0])
    duplicate.id = "opening_manual_duplicate"
    model.openings.append(duplicate)

    issues = validate_model(model)

    assert any(issue.code == "opening.duplicate_overlap" for issue in issues)


def test_model_json_round_trip_is_lossless():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    encoded = to_json_dict(model)
    json.dumps(encoded)
    decoded = from_json_dict(encoded)

    assert to_json_dict(decoded) == encoded


def test_legacy_schema_remains_unchanged():
    legacy = legacy_serialize.to_json_dict(cv_result())

    assert legacy["schema_version"] == "1.0.0"
    assert "scale" not in legacy
    assert "review_status" not in legacy["walls"][0]

