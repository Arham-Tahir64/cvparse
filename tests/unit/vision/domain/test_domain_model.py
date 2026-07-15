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
    DomainCommandError,
    move_wall_endpoint,
    redo_last_edit,
    set_approval_status,
    set_review_status,
    set_scale,
    undo_last_edit,
)
from vision.domain.geometry import distance, point_at_offset, polygon_area
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
