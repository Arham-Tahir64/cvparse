"""Phase 1 editable-domain import, validation, and serialization tests."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import fitz

from vision.adapters.domain_pdf import render_reviewed_pdf
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
from vision.adapters.domain_annotation_adapter import (
    to_model_annotation_document,
    to_model_svg,
)
from vision.domain.import_cv import import_cv_result
from vision.domain.commands import (
    add_opening,
    add_wall,
    delete_opening,
    delete_wall,
    DomainCommandError,
    move_wall_endpoint,
    recompute_room_topology,
    redo_last_edit,
    set_approval_status,
    set_opening_kind,
    set_review_status,
    set_scale,
    split_wall,
    undo_last_edit,
    update_opening_geometry,
)
from vision.domain.geometry import distance, point_at_offset, polygon_area
from vision.domain.costs import (
    EstimateAssumptions,
    MaterialEstimateError,
    calculate_material_estimate,
)
from vision.domain.models import (
    ApprovalStatus,
    Coordinate,
    ObjectSourceKind,
    OpeningKind,
    ReviewStatus,
)
from vision.domain.repository import (
    InMemoryModelRepository,
    JsonFileModelRepository,
    RevisionConflictError,
)
from vision.domain.quantities import QuantityBasis, calculate_quantities
from vision.domain.serialize import from_json_dict, to_json_dict
from vision.domain.source_assets import (
    FileSourceAssetRepository,
    SourceAssetIntegrityError,
)
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
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )

    encoded = to_json_dict(model)
    json.dumps(encoded)
    decoded = from_json_dict(encoded)

    assert to_json_dict(decoded) == encoded


def test_scale_command_recomputes_only_calibrated_measurements():
    original = import_cv_result(cv_result(), source_fingerprint="abc123")

    updated = set_scale(original, pixels_per_unit=20, unit="ft", actor="reviewer")

    assert original.revision == 1
    assert original.walls[0].length is None
    assert updated.revision == 2
    assert updated.scale.review_status == ReviewStatus.CONFIRMED
    assert updated.walls[0].length == updated.walls[0].length_px / 20
    assert updated.openings[0].width == updated.openings[0].width_px / 20
    assert updated.rooms[0].area == updated.rooms[0].area_px / 400
    assert updated.rooms[0].perimeter == updated.rooms[0].perimeter_px / 20
    assert updated.edit_history[-1].action == "set_scale"


def test_review_command_locks_confirmed_object_and_preserves_original():
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall_id = original.walls[0].id

    updated = set_review_status(
        original, object_id=wall_id, status=ReviewStatus.CONFIRMED,
    )

    assert original.walls[0].metadata.review_status != ReviewStatus.CONFIRMED
    assert updated.revision == 2
    assert updated.walls[0].metadata.review_status == ReviewStatus.CONFIRMED
    assert updated.walls[0].metadata.locked is True
    assert updated.edit_history[-1].affected_object_ids == [wall_id]


def test_json_repository_persists_and_rejects_stale_revision(tmp_path):
    repository = JsonFileModelRepository(tmp_path)
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    repository.save(original)

    loaded = repository.get(original.id)
    assert to_json_dict(loaded) == to_json_dict(original)

    updated = set_scale(loaded, pixels_per_unit=20, unit="ft")
    repository.save(updated, expected_revision=1)
    assert repository.get(original.id).revision == 2
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []

    stale = set_review_status(
        original, object_id=original.walls[0].id,
        status=ReviewStatus.CONFIRMED,
    )
    try:
        repository.save(stale, expected_revision=1)
    except RevisionConflictError:
        pass
    else:
        raise AssertionError("stale write was not rejected")


def test_undo_redo_restores_full_state_and_appends_audit_events():
    repository = InMemoryModelRepository()
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    repository.save(original)
    scaled = set_scale(original, pixels_per_unit=20, unit="ft", actor="scale-user")
    repository.save(scaled, expected_revision=1)
    wall_id = scaled.walls[0].id
    reviewed = set_review_status(
        scaled, object_id=wall_id, status=ReviewStatus.CONFIRMED,
        actor="review-user",
    )
    repository.save(reviewed, expected_revision=2)

    assert reviewed.undo_revision_stack == [1, 2]
    assert reviewed.redo_revision_stack == []
    undone = undo_last_edit(
        reviewed,
        repository.get_revision(reviewed.id, reviewed.undo_revision_stack[-1]),
        actor="undo-user",
    )
    repository.save(undone, expected_revision=3)

    undone_wall = next(wall for wall in undone.walls if wall.id == wall_id)
    assert undone.revision == 4
    assert undone_wall.metadata.review_status == ReviewStatus.LIKELY_CORRECT
    assert undone.scale.pixels_per_unit == 20
    assert undone.undo_revision_stack == [1]
    assert undone.redo_revision_stack == [3]
    assert [event.action for event in undone.edit_history] == [
        "set_scale", "set_review_status", "undo",
    ]
    assert undone.edit_history[-1].affected_object_ids == [wall_id]
    assert undone.edit_history[-1].payload["restored_snapshot_revision"] == 2
    assert repository.get_revision(reviewed.id, 3).walls[0].metadata.review_status == (
        ReviewStatus.CONFIRMED
    )

    redone = redo_last_edit(
        undone,
        repository.get_revision(undone.id, undone.redo_revision_stack[-1]),
        actor="redo-user",
    )
    repository.save(redone, expected_revision=4)

    redone_wall = next(wall for wall in redone.walls if wall.id == wall_id)
    assert redone.revision == 5
    assert redone_wall.metadata.review_status == ReviewStatus.CONFIRMED
    assert redone.undo_revision_stack == [1, 4]
    assert redone.redo_revision_stack == []
    assert [event.action for event in redone.edit_history] == [
        "set_scale", "set_review_status", "undo", "redo",
    ]
    assert redone.edit_history[-1].affected_object_ids == [wall_id]
    assert redone.edit_history[-1].payload["restored_snapshot_revision"] == 3


def test_new_edit_after_undo_invalidates_redo_branch():
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    scaled = set_scale(original, pixels_per_unit=20, unit="ft")
    reviewed = set_review_status(
        scaled, object_id=scaled.walls[0].id, status=ReviewStatus.CONFIRMED,
    )
    undone = undo_last_edit(reviewed, scaled)

    branched = set_scale(undone, pixels_per_unit=21, unit="ft")

    assert undone.redo_revision_stack == [3]
    assert branched.undo_revision_stack == [1, 4]
    assert branched.redo_revision_stack == []
    try:
        redo_last_edit(branched, reviewed)
    except DomainCommandError as exc:
        assert "nothing to redo" in str(exc)
    else:
        raise AssertionError("redo survived a new branch edit")


def test_json_revision_history_survives_repository_restart(tmp_path):
    repository = JsonFileModelRepository(tmp_path)
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    repository.save(original)
    scaled = set_scale(original, pixels_per_unit=20, unit="ft")
    repository.save(scaled, expected_revision=1)
    reviewed = set_review_status(
        scaled, object_id=scaled.walls[0].id, status=ReviewStatus.CONFIRMED,
    )
    repository.save(reviewed, expected_revision=2)

    restarted = JsonFileModelRepository(tmp_path)
    assert restarted.get(original.id).revision == 3
    assert [restarted.get_revision(original.id, revision).revision for revision in (1, 2, 3)] == [
        1, 2, 3,
    ]
    undone = undo_last_edit(restarted.get(original.id), restarted.get_revision(original.id, 2))
    restarted.save(undone, expected_revision=3)

    restarted_again = JsonFileModelRepository(tmp_path)
    assert restarted_again.get(original.id).revision == 4
    assert restarted_again.get_revision(original.id, 3).walls[0].metadata.review_status == (
        ReviewStatus.CONFIRMED
    )
    assert restarted_again.get(original.id).walls[0].metadata.review_status == (
        ReviewStatus.LIKELY_CORRECT
    )
    assert list(tmp_path.rglob("*.tmp")) == []


def test_move_shared_wall_endpoint_recomputes_only_graph_dependents():
    original = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    shared = next(node for node in original.nodes if len(node.connected_wall_ids) == 2)
    selected = next(wall for wall in original.walls if wall.end_node_id == shared.id)
    unchanged_node = next(node for node in original.nodes if node.id != shared.id)
    old_unchanged_point = unchanged_node.point

    updated = move_wall_endpoint(
        original, wall_id=selected.id, endpoint="end",
        point=Coordinate(240, 40), actor="reviewer",
    )

    moved_node = next(node for node in updated.nodes if node.id == shared.id)
    assert original.revision == 2
    assert original.nodes != updated.nodes
    assert moved_node.point == Coordinate(240, 40)
    assert next(node for node in updated.nodes if node.id == unchanged_node.id).point == old_unchanged_point
    affected_walls = [wall for wall in updated.walls if wall.id in shared.connected_wall_ids]
    assert len(affected_walls) == 2
    assert all(
        wall.start == moved_node.point or wall.end == moved_node.point
        for wall in affected_walls
    )
    assert all(wall.length == wall.length_px / 20 for wall in affected_walls)
    assert all(len(wall.polygon) == 4 for wall in affected_walls)
    assert all(
        wall.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
        and wall.metadata.review_status == ReviewStatus.NEEDS_REVIEW
        for wall in affected_walls
    )
    assert updated.revision == 3
    assert updated.edit_history[-1].action == "move_wall_endpoint"
    assert len(updated.edit_history[-1].affected_object_ids) == 8
    assert unchanged_node.id not in updated.edit_history[-1].affected_object_ids


def test_move_wall_endpoint_keeps_openings_and_symbols_attached():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    shared = next(node for node in model.nodes if len(node.connected_wall_ids) == 2)
    selected = next(wall for wall in model.walls if wall.end_node_id == shared.id)
    old_door_hinge = model.doors[0].hinge

    updated = move_wall_endpoint(
        model, wall_id=selected.id, endpoint="end", point=Coordinate(240, 40),
    )

    for wall in updated.walls:
        for opening_id in wall.opening_ids:
            opening = next(item for item in updated.openings if item.id == opening_id)
            expected = point_at_offset(
                wall.start, wall.end,
                (opening.start_offset_px + opening.end_offset_px) / 2,
            )
            assert distance(opening.center, expected) < 1e-6
            assert opening.orientation == wall.orientation
            assert opening.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert updated.doors[0].hinge != old_door_hinge
    assert updated.doors[0].metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert updated.windows[0].metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED


def test_move_wall_endpoint_updates_room_corner_area_and_perimeter():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    shared = next(node for node in model.nodes if len(node.connected_wall_ids) == 2)
    selected = next(wall for wall in model.walls if wall.end_node_id == shared.id)

    updated = move_wall_endpoint(
        model, wall_id=selected.id, endpoint="end", point=Coordinate(240, 40),
    )

    room = updated.rooms[0]
    assert Coordinate(240, 40) in room.polygon
    assert Coordinate(220, 20) not in room.polygon
    assert room.area_px == polygon_area(room.polygon)
    assert room.area == room.area_px / 400
    assert room.perimeter == room.perimeter_px / 20
    assert room.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED


def test_move_wall_endpoint_rejects_geometry_that_orphans_opening():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall = model.walls[0]

    try:
        move_wall_endpoint(
            model, wall_id=wall.id, endpoint="end", point=Coordinate(90, 20),
        )
    except DomainCommandError as exc:
        assert "opening beyond wall" in str(exc)
    else:
        raise AssertionError("invalid shortening was accepted")
    assert wall.end == Coordinate(220, 20)


def test_add_wall_snaps_nodes_updates_graph_quantities_and_invalidates_room():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    reused_node = next(node for node in model.nodes if node.point == Coordinate(20, 20))
    incident_wall = next(
        wall for wall in model.walls if reused_node.id in {
            wall.start_node_id, wall.end_node_id,
        }
    )
    before_quantities = calculate_quantities(model)

    updated = add_wall(
        model,
        start=Coordinate(21, 21),
        end=Coordinate(20, 220),
        thickness_px=12,
        wall_type="interior",
        actor="wall-editor",
    )

    event = updated.edit_history[-1]
    added = next(wall for wall in updated.walls if wall.id == event.payload["wall_id"])
    updated_incident = next(wall for wall in updated.walls if wall.id == incident_wall.id)
    assert event.action == "add_wall"
    assert updated.revision == model.revision + 1
    assert added.start == Coordinate(20, 20)
    assert added.start_node_id == reused_node.id
    assert added.length_px == 200
    assert added.length == 10
    assert added.wall_type == "interior"
    assert added.metadata.source.kind == ObjectSourceKind.MANUAL_CREATED
    assert incident_wall.id in added.connected_wall_ids
    assert added.id in updated_incident.connected_wall_ids
    assert updated_incident.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert len(updated.nodes) == len(model.nodes) + 1
    assert calculate_quantities(updated).pixel_measurements[
        "wall_centerline_length_px"
    ] == before_quantities.pixel_measurements["wall_centerline_length_px"] + 200
    assert updated.rooms[0].metadata.review_status == ReviewStatus.NEEDS_REVIEW
    assert any(
        issue.code == "room.topology_stale" for issue in updated.validation_issues
    )

    undone = undo_last_edit(updated, model)
    assert not any(wall.id == added.id for wall in undone.walls)
    assert undone.rooms[0].metadata.source.kind == ObjectSourceKind.AUTOMATIC_INFERRED
    assert not any(
        issue.code == "room.topology_stale" for issue in undone.validation_issues
    )


def test_add_wall_rejects_duplicate_and_unsplit_crossing():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    for start, end, expected in (
        (Coordinate(21, 20), Coordinate(219, 20), "duplicates existing wall"),
        (Coordinate(120, 0), Coordinate(120, 80), "without a shared endpoint"),
        (Coordinate(120, 20), Coordinate(120, 100), "without a shared endpoint"),
        (Coordinate(20, 20), Coordinate(100, 20), "without a shared endpoint"),
    ):
        try:
            add_wall(model, start=start, end=end, thickness_px=12)
        except DomainCommandError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid wall creation was accepted")
    assert model.revision == 1
    assert model.undo_revision_stack == []


def test_split_wall_preserves_parent_id_reassigns_opening_and_keeps_quantities():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    wall = model.walls[1]
    opening = next(item for item in model.openings if item.wall_id == wall.id)
    window = next(item for item in model.windows if item.opening_id == opening.id)
    model.rooms[0].boundary_wall_ids = [wall.id]
    model.rooms[0].metadata.review_status = ReviewStatus.CONFIRMED
    model.rooms[0].metadata.locked = True
    before_quantities = calculate_quantities(model)

    updated = split_wall(
        model,
        wall_id=wall.id,
        point=Coordinate(218, 80),
        projection_tolerance_px=5,
        actor="wall-editor",
    )

    event = updated.edit_history[-1]
    parent = next(item for item in updated.walls if item.id == wall.id)
    child = next(
        item for item in updated.walls if item.id == event.payload["new_wall_id"]
    )
    split_node = next(
        item for item in updated.nodes if item.id == event.payload["split_node_id"]
    )
    moved_opening = next(item for item in updated.openings if item.id == opening.id)
    moved_window = next(item for item in updated.windows if item.id == window.id)
    after_quantities = calculate_quantities(updated)

    assert event.action == "split_wall"
    assert event.payload["projected_point"] == {"x": 220.0, "y": 80.0}
    assert parent.id == wall.id
    assert parent.end == Coordinate(220, 80)
    assert parent.length_px == 60
    assert child.start == Coordinate(220, 80)
    assert child.end == Coordinate(220, 220)
    assert child.length_px == 140
    assert child.metadata.source.kind == ObjectSourceKind.MANUAL_CREATED
    assert child.metadata.source.details["parent_wall_id"] == wall.id
    assert split_node.point == Coordinate(220, 80)
    assert set(split_node.connected_wall_ids) == {wall.id, child.id}
    assert moved_opening.wall_id == child.id
    assert moved_opening.start_offset_px == 15
    assert moved_opening.end_offset_px == 45
    assert moved_opening.center == opening.center
    assert moved_opening.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert moved_window.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert updated.rooms[0].boundary_wall_ids == [wall.id, child.id]
    assert updated.rooms[0].metadata.review_status == ReviewStatus.CONFIRMED
    assert updated.rooms[0].metadata.locked is True
    assert after_quantities.pixel_measurements == before_quantities.pixel_measurements
    assert not any(
        issue.code == "room.topology_stale" for issue in updated.validation_issues
    )

    undone = undo_last_edit(updated, model)
    assert len(undone.walls) == len(model.walls)
    restored_opening = next(item for item in undone.openings if item.id == opening.id)
    assert restored_opening.wall_id == wall.id
    redone = redo_last_edit(undone, updated)
    assert any(item.id == child.id for item in redone.walls)


def test_split_wall_rejects_opening_crossing_and_distant_click():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall = model.walls[0]

    for point, tolerance, expected in (
        (Coordinate(100, 20), None, "crosses opening"),
        (Coordinate(150, 100), 10, "beyond tolerance"),
        (Coordinate(20, 20), None, "inside both wall endpoints"),
    ):
        try:
            split_wall(
                model,
                wall_id=wall.id,
                point=point,
                projection_tolerance_px=tolerance,
            )
        except DomainCommandError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid wall split was accepted")
    assert model.revision == 1


def test_split_wall_enables_connected_t_junction_creation():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall = model.walls[0]
    split = split_wall(
        model, wall_id=wall.id, point=Coordinate(150, 20),
    )
    split_node_id = split.edit_history[-1].payload["split_node_id"]
    before_length = calculate_quantities(split).pixel_measurements[
        "wall_centerline_length_px"
    ]

    joined = add_wall(
        split,
        start=Coordinate(150, 20),
        end=Coordinate(150, 120),
        thickness_px=12,
    )

    split_node = next(node for node in joined.nodes if node.id == split_node_id)
    added_wall_id = joined.edit_history[-1].payload["wall_id"]
    assert len(split_node.connected_wall_ids) == 3
    assert added_wall_id in split_node.connected_wall_ids
    assert calculate_quantities(joined).pixel_measurements[
        "wall_centerline_length_px"
    ] == before_length + 100
    assert any(
        issue.code == "room.topology_stale" for issue in joined.validation_issues
    )


def test_add_opening_projects_to_wall_creates_door_and_updates_quantities():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    wall = model.walls[0]
    model.rooms[0].boundary_wall_ids = [wall.id]
    before = calculate_quantities(model)

    updated = add_opening(
        model,
        wall_id=wall.id,
        center=Coordinate(180, 23),
        width_px=20,
        kind=OpeningKind.DOOR,
        actor="opening-editor",
    )

    event = updated.edit_history[-1]
    opening = next(
        item for item in updated.openings
        if item.id == event.payload["opening_id"]
    )
    door = next(item for item in updated.doors if item.opening_id == opening.id)
    updated_wall = next(item for item in updated.walls if item.id == wall.id)
    after = calculate_quantities(updated)
    assert event.action == "add_opening"
    assert opening.center == Coordinate(180, 20)
    assert opening.start_offset_px == 150
    assert opening.end_offset_px == 170
    assert opening.width_px == 20
    assert opening.width == 1
    assert opening.kind == OpeningKind.DOOR
    assert opening.metadata.source.kind == ObjectSourceKind.MANUAL_CREATED
    assert door.metadata.source.kind == ObjectSourceKind.MANUAL_CREATED
    assert door.hinge is None
    assert opening.id in updated_wall.opening_ids
    assert door.id in updated.rooms[0].door_ids
    assert updated.rooms[0].metadata.review_status == ReviewStatus.NEEDS_REVIEW
    assert after.counts["openings"] == before.counts["openings"] + 1
    assert after.counts["doors"] == before.counts["doors"] + 1
    assert after.pixel_measurements["opening_width_px"] == (
        before.pixel_measurements["opening_width_px"] + 20
    )
    annotation = to_model_annotation_document(updated)
    assert any(item["id"] == door.id for item in annotation["elements"])

    undone = undo_last_edit(updated, model)
    assert not any(item.id == opening.id for item in undone.openings)
    assert not any(item.id == door.id for item in undone.doors)


def test_add_opening_rejects_overlap_off_wall_and_out_of_bounds():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall = model.walls[0]

    for center, width, tolerance, expected in (
        (Coordinate(100, 20), 20, None, "overlaps existing opening"),
        (Coordinate(180, 80), 20, 10, "beyond tolerance"),
        (Coordinate(215, 20), 30, None, "fully within"),
    ):
        try:
            add_opening(
                model,
                wall_id=wall.id,
                center=center,
                width_px=width,
                projection_tolerance_px=tolerance,
            )
        except DomainCommandError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid opening creation was accepted")
    assert model.revision == 1


def test_update_opening_geometry_moves_door_swing_resizes_and_is_undoable():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    opening = next(item for item in model.openings if item.kind == OpeningKind.DOOR)
    door = next(item for item in model.doors if item.opening_id == opening.id)
    before = calculate_quantities(model)

    updated = update_opening_geometry(
        model,
        opening_id=opening.id,
        center=Coordinate(150, 20),
        width_px=30,
        actor="opening-editor",
    )

    moved = next(item for item in updated.openings if item.id == opening.id)
    moved_door = next(item for item in updated.doors if item.id == door.id)
    after = calculate_quantities(updated)
    assert updated.edit_history[-1].action == "update_opening_geometry"
    assert moved.center == Coordinate(150, 20)
    assert moved.start_offset_px == 115
    assert moved.end_offset_px == 145
    assert moved.width_px == 30
    assert moved.width == 1.5
    assert moved.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert moved_door.hinge == Coordinate(150, 20)
    assert moved_door.swing_end == Coordinate(150, 50)
    assert moved_door.swing_arc == [
        Coordinate(
            150 + (point.x - door.hinge.x) * 0.75,
            20 + (point.y - door.hinge.y) * 0.75,
        )
        for point in door.swing_arc
    ]
    assert after.pixel_measurements["opening_width_px"] == (
        before.pixel_measurements["opening_width_px"] - 10
    )

    undone = undo_last_edit(updated, model)
    restored = next(item for item in undone.openings if item.id == opening.id)
    restored_door = next(item for item in undone.doors if item.id == door.id)
    assert restored.center == opening.center
    assert restored.width_px == opening.width_px
    assert restored_door.swing_arc == door.swing_arc


def test_update_opening_geometry_rejects_noop_and_other_opening_overlap():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    opening = next(item for item in model.openings if item.kind == OpeningKind.DOOR)
    with_extra = add_opening(
        model,
        wall_id=opening.wall_id,
        center=Coordinate(180, 20),
        width_px=20,
    )

    for source, center, width, expected in (
        (model, opening.center, opening.width_px, "did not change"),
        (with_extra, Coordinate(180, 20), 30, "overlaps existing opening"),
    ):
        try:
            update_opening_geometry(
                source,
                opening_id=opening.id,
                center=center,
                width_px=width,
            )
        except DomainCommandError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid opening geometry update was accepted")


def test_opening_reclassification_preserves_physical_id_and_migrates_relations():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    opening = next(item for item in model.openings if item.kind == OpeningKind.DOOR)
    door = next(item for item in model.doors if item.opening_id == opening.id)
    room = model.rooms[0]
    room.boundary_wall_ids = [opening.wall_id]
    room.door_ids = [door.id]
    before = calculate_quantities(model)

    updated = set_opening_kind(
        model,
        opening_id=opening.id,
        kind=OpeningKind.WINDOW,
        actor="opening-editor",
    )

    converted = next(item for item in updated.openings if item.id == opening.id)
    new_window = next(item for item in updated.windows if item.opening_id == opening.id)
    after = calculate_quantities(updated)
    event = updated.edit_history[-1]
    assert event.action == "set_opening_kind"
    assert converted.id == opening.id
    assert converted.kind == OpeningKind.WINDOW
    assert converted.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert not any(item.id == door.id for item in updated.doors)
    assert new_window.metadata.source.kind == ObjectSourceKind.MANUAL_CREATED
    assert door.id not in updated.rooms[0].door_ids
    assert new_window.id in updated.rooms[0].window_ids
    assert updated.rooms[0].metadata.review_status == ReviewStatus.NEEDS_REVIEW
    assert after.counts["openings"] == before.counts["openings"]
    assert after.counts["doors"] == before.counts["doors"] - 1
    assert after.counts["windows"] == before.counts["windows"] + 1
    assert after.pixel_measurements["opening_width_px"] == (
        before.pixel_measurements["opening_width_px"]
    )
    annotations = to_model_annotation_document(updated)
    assert any(item["id"] == new_window.id for item in annotations["elements"])
    assert not any(item["id"] == door.id for item in annotations["elements"])

    undone = undo_last_edit(updated, model)
    assert any(item.id == door.id for item in undone.doors)
    assert not any(item.id == new_window.id for item in undone.windows)
    restored = next(item for item in undone.openings if item.id == opening.id)
    assert restored.kind == OpeningKind.DOOR
    redone = redo_last_edit(undone, updated)
    assert any(item.id == new_window.id for item in redone.windows)


def test_opening_reclassification_rejects_noop_without_mutation():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    opening = model.openings[0]

    try:
        set_opening_kind(model, opening_id=opening.id, kind=opening.kind)
    except DomainCommandError as exc:
        assert "already classified" in str(exc)
    else:
        raise AssertionError("no-op opening classification was accepted")
    assert model.revision == 1
    assert model.undo_revision_stack == []


def test_delete_opening_requires_cascade_and_restores_dependency_chain_on_undo():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    opening = next(item for item in model.openings if item.kind == OpeningKind.DOOR)
    door = next(item for item in model.doors if item.opening_id == opening.id)
    wall = next(item for item in model.walls if item.id == opening.wall_id)
    model.rooms[0].boundary_wall_ids = [wall.id]
    model.rooms[0].door_ids = [door.id]
    before = calculate_quantities(model)

    try:
        delete_opening(model, opening_id=opening.id)
    except DomainCommandError as exc:
        assert "dependent objects" in str(exc)
        assert "cascade=true" in str(exc)
    else:
        raise AssertionError("opening dependency deletion did not require cascade")

    deleted = delete_opening(
        model, opening_id=opening.id, cascade=True, actor="opening-editor",
    )

    after = calculate_quantities(deleted)
    deleted_wall = next(item for item in deleted.walls if item.id == wall.id)
    assert not any(item.id == opening.id for item in deleted.openings)
    assert not any(item.id == door.id for item in deleted.doors)
    assert opening.id not in deleted_wall.opening_ids
    assert door.id not in deleted.rooms[0].door_ids
    assert after.counts["openings"] == before.counts["openings"] - 1
    assert after.counts["doors"] == before.counts["doors"] - 1
    assert after.pixel_measurements["opening_width_px"] == (
        before.pixel_measurements["opening_width_px"] - opening.width_px
    )
    assert deleted.edit_history[-1].action == "delete_opening"
    assert deleted.edit_history[-1].payload["cascade"] is True

    restored = undo_last_edit(deleted, model)
    assert any(item.id == opening.id for item in restored.openings)
    assert any(item.id == door.id for item in restored.doors)
    assert opening.id in next(
        item for item in restored.walls if item.id == wall.id
    ).opening_ids


def test_delete_unclassified_opening_needs_no_cascade():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    added = add_opening(
        model,
        wall_id=model.walls[0].id,
        center=Coordinate(180, 20),
        width_px=20,
        kind=OpeningKind.UNKNOWN,
    )
    opening_id = added.edit_history[-1].payload["opening_id"]

    deleted = delete_opening(added, opening_id=opening_id)

    assert not any(item.id == opening_id for item in deleted.openings)
    assert deleted.edit_history[-1].payload["cascade"] is False


def _closed_room_model():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    bottom = add_wall(
        model,
        start=Coordinate(220, 220),
        end=Coordinate(20, 220),
        thickness_px=12,
    )
    return add_wall(
        bottom,
        start=Coordinate(20, 220),
        end=Coordinate(20, 20),
        thickness_px=12,
    )


def test_room_topology_recompute_recovers_face_relations_and_stable_id():
    closed = _closed_room_model()
    original_room_id = closed.rooms[0].id
    assert any(
        issue.code == "room.topology_stale" for issue in closed.validation_issues
    )

    updated = recompute_room_topology(closed, actor="room-editor")

    assert updated.revision == closed.revision + 1
    assert len(updated.rooms) == 1
    room = updated.rooms[0]
    assert room.id == original_room_id
    assert room.label == "OFFICE"
    assert room.area_px == 40000
    assert room.perimeter_px == 800
    assert set(room.boundary_wall_ids) == {wall.id for wall in updated.walls}
    assert room.door_ids == [updated.doors[0].id]
    assert room.window_ids == [updated.windows[0].id]
    assert room.metadata.source.kind == ObjectSourceKind.MANUAL_ADJUSTED
    assert room.metadata.source.stage == "room_topology_recompute"
    assert room.metadata.review_status == ReviewStatus.NEEDS_REVIEW
    assert not any(
        issue.code == "room.topology_stale" for issue in updated.validation_issues
    )
    assert updated.edit_history[-1].action == "recompute_room_topology"
    assert updated.edit_history[-1].payload["matched_room_ids"] == [original_room_id]
    assert calculate_quantities(updated).pixel_measurements["floor_area_px"] == 40000

    undone = undo_last_edit(updated, closed)
    assert any(
        issue.code == "room.topology_stale" for issue in undone.validation_issues
    )
    redone = redo_last_edit(undone, updated)
    assert redone.rooms[0].id == original_room_id
    assert not any(
        issue.code == "room.topology_stale" for issue in redone.validation_issues
    )


def test_room_topology_recompute_splits_face_and_builds_adjacency():
    room_model = recompute_room_topology(_closed_room_model())
    top = next(
        wall for wall in room_model.walls
        if wall.start.y == wall.end.y == 20
    )
    bottom = next(
        wall for wall in room_model.walls
        if wall.start.y == wall.end.y == 220
    )
    split_top = split_wall(
        room_model, wall_id=top.id, point=Coordinate(120, 20),
    )
    split_bottom = split_wall(
        split_top, wall_id=bottom.id, point=Coordinate(120, 220),
    )
    top_node = split_top.edit_history[-1].payload["split_node_id"]
    bottom_node = split_bottom.edit_history[-1].payload["split_node_id"]
    top_point = next(node.point for node in split_bottom.nodes if node.id == top_node)
    bottom_point = next(node.point for node in split_bottom.nodes if node.id == bottom_node)
    divided = add_wall(
        split_bottom,
        start=top_point,
        end=bottom_point,
        thickness_px=12,
    )
    divider_id = divided.edit_history[-1].payload["wall_id"]

    updated = recompute_room_topology(divided)

    assert len(updated.rooms) == 2
    assert sorted(room.area_px for room in updated.rooms) == [20000, 20000]
    assert sum(room.area_px for room in updated.rooms) == 40000
    assert all(divider_id in room.boundary_wall_ids for room in updated.rooms)
    assert all(len(room.neighboring_room_ids) == 1 for room in updated.rooms)
    assert updated.rooms[0].neighboring_room_ids == [updated.rooms[1].id]
    assert updated.rooms[1].neighboring_room_ids == [updated.rooms[0].id]
    assert sum(room.label == "OFFICE" for room in updated.rooms) == 1
    assert len(updated.edit_history[-1].payload["created_room_ids"]) == 1
    assert not any(
        issue.code == "room.topology_stale" for issue in updated.validation_issues
    )
    assert calculate_quantities(updated).pixel_measurements["floor_area_px"] == 40000

    without_divider = delete_wall(updated, wall_id=divider_id, cascade=True)
    merged = recompute_room_topology(without_divider)

    assert len(merged.rooms) == 1
    assert merged.rooms[0].area_px == 40000
    assert merged.rooms[0].neighboring_room_ids == []
    assert divider_id not in merged.rooms[0].boundary_wall_ids
    assert len(merged.edit_history[-1].payload["removed_room_ids"]) == 1
    assert calculate_quantities(merged).pixel_measurements["floor_area_px"] == 40000


def test_delete_manual_wall_repairs_graph_and_is_undoable():
    original = import_cv_result(cv_result(), source_fingerprint="abc123")
    added = add_wall(
        original,
        start=Coordinate(20, 20),
        end=Coordinate(20, 220),
        thickness_px=12,
    )
    added_id = added.edit_history[-1].payload["wall_id"]
    added_wall = next(wall for wall in added.walls if wall.id == added_id)
    new_node_id = added_wall.end_node_id

    deleted = delete_wall(added, wall_id=added_id, actor="wall-editor")

    assert deleted.edit_history[-1].action == "delete_wall"
    assert not any(wall.id == added_id for wall in deleted.walls)
    assert not any(node.id == new_node_id for node in deleted.nodes)
    assert all(
        added_id not in wall.connected_wall_ids for wall in deleted.walls
    )
    assert not any(
        issue.code == "room.topology_stale" for issue in deleted.validation_issues
    )
    assert calculate_quantities(deleted).counts["walls"] == len(original.walls)

    restored = undo_last_edit(deleted, added)
    assert any(wall.id == added_id for wall in restored.walls)
    assert restored.edit_history[-1].action == "undo"


def test_delete_wall_requires_explicit_cascade_for_opening_dependencies():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    wall = model.walls[0]
    opening_ids = set(wall.opening_ids)
    door_ids = {
        door.id for door in model.doors if door.opening_id in opening_ids
    }

    try:
        delete_wall(model, wall_id=wall.id)
    except DomainCommandError as exc:
        assert "dependent objects" in str(exc)
        assert "cascade=true" in str(exc)
    else:
        raise AssertionError("dependent wall deletion did not require cascade")
    assert any(item.id == wall.id for item in model.walls)

    deleted = delete_wall(model, wall_id=wall.id, cascade=True)

    assert not any(item.id == wall.id for item in deleted.walls)
    assert not opening_ids.intersection(item.id for item in deleted.openings)
    assert not door_ids.intersection(item.id for item in deleted.doors)
    assert all(
        wall.id not in item.connected_wall_ids for item in deleted.walls
    )
    assert any(
        issue.code == "room.topology_stale" for issue in deleted.validation_issues
    )
    assert deleted.edit_history[-1].payload["cascade"] is True


def test_validation_flags_wall_endpoint_that_disagrees_with_shared_node():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    model.walls[0].end = Coordinate(219, 19)

    issues = validate_model(model)

    mismatch = next(issue for issue in issues if issue.code == "wall.node_geometry_mismatch")
    assert model.walls[0].id in mismatch.affected_object_ids


def test_model_annotation_export_uses_current_reviewed_geometry():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    shared = next(node for node in model.nodes if len(node.connected_wall_ids) == 2)
    wall = next(item for item in model.walls if item.end_node_id == shared.id)
    moved = move_wall_endpoint(
        model, wall_id=wall.id, endpoint="end", point=Coordinate(240, 40),
    )
    rejected_wall_id = moved.walls[1].id
    reviewed = set_review_status(
        moved, object_id=rejected_wall_id, status=ReviewStatus.REJECTED,
    )

    document = to_model_annotation_document(reviewed)
    exported_wall = next(
        item for item in document["elements"] if item["id"] == wall.id
    )

    assert document["model_revision"] == reviewed.revision
    assert exported_wall["geometry"]["centerline"]["x2"] == 240
    assert exported_wall["geometry"]["centerline"]["y2"] == 40
    assert exported_wall["review_state"] == "needs_review"
    assert not any(item["id"] == rejected_wall_id for item in document["elements"])


def test_model_svg_is_deterministic_and_reflects_manual_revision():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    shared = next(node for node in model.nodes if len(node.connected_wall_ids) == 2)
    wall = next(item for item in model.walls if item.end_node_id == shared.id)
    updated = move_wall_endpoint(
        model, wall_id=wall.id, endpoint="end", point=Coordinate(240, 40),
    )

    first = to_model_svg(updated)
    second = to_model_svg(from_json_dict(to_json_dict(updated)))

    ET.fromstring(first)
    assert first == second
    assert f'data-model-revision="{updated.revision}"' in first
    assert f'data-id="{wall.id}"' in first
    assert "240,40" in first
    assert 'data-source="manual_adjusted"' in first


def test_provisional_quantities_are_explicitly_unscaled_and_non_authoritative():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    summary = calculate_quantities(model)

    assert summary.basis == QuantityBasis.PROVISIONAL
    assert summary.counts == {
        "walls": 2, "openings": 2, "doors": 1, "windows": 1, "rooms": 1,
    }
    assert summary.pixel_measurements["wall_centerline_length_px"] == 400
    assert summary.calibrated_measurements["floor_area"] is None
    assert summary.authoritative is False
    assert any("automatic candidates" in warning for warning in summary.warnings)


def test_verified_quantities_require_confirmed_dependency_chain_and_scale():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    for collection in (
        model.walls, model.openings, model.rooms, model.doors, model.windows,
    ):
        for item in collection:
            item.metadata.review_status = ReviewStatus.CONFIRMED
    model.validation_issues = validate_model(model)

    summary = calculate_quantities(model, QuantityBasis.VERIFIED)

    assert summary.counts["doors"] == 1
    assert summary.counts["windows"] == 1
    assert summary.calibrated_measurements["wall_centerline_length"] == 20
    assert summary.calibrated_measurements["floor_area"] == 100
    assert summary.calibrated_measurements["opening_width"] == 3.5
    assert summary.complete is True
    assert summary.authoritative is True


def test_verified_quantities_exclude_door_without_confirmed_opening():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    door = model.doors[0]
    opening = next(item for item in model.openings if item.id == door.opening_id)
    wall = next(item for item in model.walls if item.id == opening.wall_id)
    door.metadata.review_status = ReviewStatus.CONFIRMED
    wall.metadata.review_status = ReviewStatus.CONFIRMED

    summary = calculate_quantities(model, QuantityBasis.VERIFIED)

    assert summary.counts["doors"] == 0
    assert door.id in summary.excluded_object_ids
    assert summary.complete is False
    assert any("dependencies" in warning for warning in summary.warnings)


def _priced_assumptions(**overrides):
    values = {
        "wall_height": 8,
        "door_height": 7,
        "window_height": 4,
        "stud_spacing": 2,
        "waste_factors": {"flooring": 0.1},
        "unit_costs": {
            "drywall": 1,
            "paint": 0.5,
            "insulation": 0.75,
            "framing_lumber": 2,
            "flooring": 3,
            "ceiling": 1,
            "baseboard": 2,
            "doors": 200,
            "windows": 300,
            "glazing": 10,
            "door_trim": 1,
            "window_trim": 1,
        },
    }
    values.update(overrides)
    return EstimateAssumptions(**values)


def test_material_estimate_applies_opening_deductions_waste_and_rates():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    for collection in (
        model.walls, model.openings, model.rooms, model.doors, model.windows,
    ):
        for item in collection:
            item.metadata.review_status = ReviewStatus.CONFIRMED
    model.rooms[0].door_ids = [model.doors[0].id]
    model.validation_issues = validate_model(model)

    estimate = calculate_material_estimate(
        model, _priced_assumptions(), QuantityBasis.VERIFIED,
    )
    lines = {line.code: line for line in estimate.line_items}

    assert lines["drywall"].quantity == 280
    assert lines["insulation"].quantity == 140
    assert lines["framing_lumber"].quantity == 188
    assert lines["flooring"].quantity == 100
    assert lines["flooring"].purchase_quantity == 110
    assert lines["flooring"].extended_cost == 330
    assert lines["baseboard"].quantity == 38
    assert lines["glazing"].quantity == 6
    assert lines["door_trim"].quantity == 32
    assert lines["window_trim"].quantity == 22
    assert estimate.priced_subtotal == 2021
    assert estimate.geometry_complete is True
    assert estimate.cost_complete is True
    assert estimate.authoritative is True
    assert set(lines["drywall"].source_object_ids).issuperset(
        opening.id for opening in model.openings
    )

    missing_relation = copy.deepcopy(model)
    missing_relation.rooms[0].door_ids = []
    incomplete = calculate_material_estimate(
        missing_relation, _priced_assumptions(), QuantityBasis.VERIFIED,
    )
    assert incomplete.geometry_complete is False
    assert incomplete.authoritative is False
    assert any("Door-room associations are missing" in item for item in incomplete.warnings)


def test_material_estimate_tracks_geometry_revision_and_unscaled_limit():
    unscaled = import_cv_result(cv_result(), source_fingerprint="abc123")
    unavailable = calculate_material_estimate(unscaled, _priced_assumptions())
    assert all(line.quantity is None for line in unavailable.line_items)
    assert unavailable.priced_subtotal == 0
    assert unavailable.cost_complete is False

    model = set_scale(unscaled, pixels_per_unit=20, unit="ft")
    before = calculate_material_estimate(model, _priced_assumptions())
    door_opening = next(
        opening for opening in model.openings if opening.kind == OpeningKind.DOOR
    )
    edited = update_opening_geometry(
        model,
        opening_id=door_opening.id,
        center=door_opening.center,
        width_px=60,
    )
    after = calculate_material_estimate(edited, _priced_assumptions())
    before_lines = {line.code: line for line in before.line_items}
    after_lines = {line.code: line for line in after.line_items}

    assert after.model_revision == edited.revision == model.revision + 1
    assert after_lines["drywall"].quantity == before_lines["drywall"].quantity - 14
    assert after_lines["baseboard"].quantity == before_lines["baseboard"].quantity - 1
    assert after_lines["ceiling"].quantity == before_lines["ceiling"].quantity

    try:
        calculate_material_estimate(
            model, _priced_assumptions(unit_costs={"made_up": 1}),
        )
    except MaterialEstimateError as exc:
        assert "unknown material codes" in str(exc)
    else:
        raise AssertionError("unknown material code was accepted")


def test_file_source_repository_is_idempotent_atomic_and_detects_tampering(tmp_path):
    content = b"original floorplan bytes"
    import hashlib

    fingerprint = hashlib.sha256(content).hexdigest()
    repository = FileSourceAssetRepository(tmp_path)

    repository.save(fingerprint, content)
    repository.save(fingerprint, content)

    assert repository.get(fingerprint) == content
    assert len(list(tmp_path.glob("*.bin"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []

    stored = next(tmp_path.glob("*.bin"))
    stored.write_bytes(b"tampered")
    try:
        repository.get(fingerprint)
    except SourceAssetIntegrityError:
        pass
    else:
        raise AssertionError("tampered source asset was accepted")


def test_reviewed_pdf_preserves_source_and_renders_current_model_revision():
    source = fitz.open()
    page = source.new_page(width=180, height=144)
    page.insert_text((10, 15), "ORIGINAL SOURCE")
    source_bytes = source.tobytes()
    source.close()

    model = import_cv_result(cv_result(), source_fingerprint="abc123")
    shared = next(node for node in model.nodes if len(node.connected_wall_ids) == 2)
    wall = next(item for item in model.walls if item.end_node_id == shared.id)
    updated = move_wall_endpoint(
        model, wall_id=wall.id, endpoint="end", point=Coordinate(240, 40),
    )

    output = render_reviewed_pdf(source_bytes, "application/pdf", updated)
    reviewed = fitz.open(stream=output, filetype="pdf")

    assert reviewed.page_count == 1
    assert "ORIGINAL SOURCE" in reviewed[0].get_text()
    assert reviewed.metadata["subject"] == f"Model {updated.id} revision {updated.revision}"
    drawings = reviewed[0].get_drawings()
    assert len(drawings) >= len(updated.walls) + len(updated.rooms)
    assert max(drawing["rect"].x1 for drawing in drawings) >= 240 * 72 / 200
    reviewed.close()


def test_approval_requires_complete_verified_takeoff():
    model = import_cv_result(cv_result(), source_fingerprint="abc123")

    try:
        set_approval_status(model, status=ApprovalStatus.APPROVED)
    except DomainCommandError as exc:
        assert "cannot be approved" in str(exc)
    else:
        raise AssertionError("unscaled automatic model was approved")
    assert model.approval_status == ApprovalStatus.DRAFT
    assert model.revision == 1


def test_approved_model_is_frozen_until_explicitly_reopened():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    for collection in (
        model.walls, model.openings, model.rooms, model.doors, model.windows,
    ):
        for item in collection:
            item.metadata.review_status = ReviewStatus.CONFIRMED
    model.validation_issues = validate_model(model)

    approved = set_approval_status(
        model, status=ApprovalStatus.APPROVED, actor="approver",
    )

    assert approved.approval_status == ApprovalStatus.APPROVED
    assert approved.revision == model.revision + 1
    assert approved.edit_history[-1].action == "set_approval_status"
    assert calculate_quantities(
        approved, QuantityBasis.VERIFIED,
    ).authoritative is True
    try:
        set_scale(approved, pixels_per_unit=21, unit="ft")
    except DomainCommandError as exc:
        assert "reopened explicitly" in str(exc)
    else:
        raise AssertionError("approved model accepted a geometry-affecting edit")

    reopened = set_approval_status(
        approved, status=ApprovalStatus.IN_REVIEW, actor="approver",
    )
    edited = set_scale(reopened, pixels_per_unit=21, unit="ft")

    assert reopened.approval_status == ApprovalStatus.IN_REVIEW
    assert edited.scale.pixels_per_unit == 21
    assert edited.revision == reopened.revision + 1


def test_approved_model_blocks_history_restore_until_reopened():
    model = set_scale(
        import_cv_result(cv_result(), source_fingerprint="abc123"),
        pixels_per_unit=20, unit="ft",
    )
    for collection in (
        model.walls, model.openings, model.rooms, model.doors, model.windows,
    ):
        for item in collection:
            item.metadata.review_status = ReviewStatus.CONFIRMED
    model.validation_issues = validate_model(model)
    approved = set_approval_status(model, status=ApprovalStatus.APPROVED)

    try:
        undo_last_edit(approved, model)
    except DomainCommandError as exc:
        assert "reopened explicitly" in str(exc)
    else:
        raise AssertionError("approved model allowed undo")

    reopened = set_approval_status(approved, status=ApprovalStatus.IN_REVIEW)
    edited = set_scale(reopened, pixels_per_unit=21, unit="ft")
    restored = undo_last_edit(edited, reopened)

    assert restored.approval_status == ApprovalStatus.IN_REVIEW
    assert restored.scale.pixels_per_unit == 20
    assert restored.edit_history[-1].action == "undo"


def test_legacy_schema_remains_unchanged():
    legacy = legacy_serialize.to_json_dict(cv_result())

    assert legacy["schema_version"] == "1.0.0"
    assert "scale" not in legacy
    assert "review_status" not in legacy["walls"][0]
