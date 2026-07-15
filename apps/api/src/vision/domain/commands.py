"""Revisioned human correction commands and targeted recomputation."""
from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone

from .geometry import (
    distance,
    point_at_offset,
    point_in_polygon,
    point_on_segment,
    polygon_area,
    polygon_perimeter,
    project_point_to_segment,
    segments_intersect,
    transform_wall_local_point,
    wall_orientation,
    wall_polygon,
)
from .models import (
    ApprovalStatus,
    ConfidenceBreakdown,
    Coordinate,
    Door,
    EditEvent,
    Node,
    ObjectMetadata,
    ObjectSourceKind,
    Opening,
    OpeningKind,
    ReviewStatus,
    ScaleMethod,
    SourceEvidence,
    TakeoffModel,
    Wall,
    Window,
)
from .validation import validate_model
from .quantities import QuantityBasis, calculate_quantities


class DomainCommandError(ValueError):
    pass


def _event(
    model: TakeoffModel,
    action: str,
    actor: str,
    revision_before: int,
    affected: list[str],
    payload: dict,
) -> EditEvent:
    return EditEvent(
        id=f"edit_{uuid.uuid4().hex}", action=action, actor=actor,
        revision_before=revision_before, revision_after=model.revision,
        affected_object_ids=sorted(set(affected)), payload=payload,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _ensure_editable(model: TakeoffModel) -> None:
    if model.approval_status == ApprovalStatus.APPROVED:
        raise DomainCommandError(
            "approved model revisions must be reopened explicitly before editing"
        )


def _prepare_edit(model: TakeoffModel) -> TakeoffModel:
    updated = copy.deepcopy(model)
    updated.undo_revision_stack.append(model.revision)
    updated.redo_revision_stack.clear()
    return updated


def set_scale(
    model: TakeoffModel,
    *,
    pixels_per_unit: float,
    unit: str,
    actor: str = "user",
) -> TakeoffModel:
    """Confirm scale and recompute only calibrated measurements."""
    _ensure_editable(model)
    if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 0:
        raise DomainCommandError("pixels_per_unit must be a positive finite number")
    if not unit.strip():
        raise DomainCommandError("unit is required")

    updated = _prepare_edit(model)
    before = updated.revision
    previous = {
        "pixels_per_unit": updated.scale.pixels_per_unit,
        "unit": updated.scale.unit,
        "method": updated.scale.method.value,
        "review_status": updated.scale.review_status.value,
    }
    updated.scale.pixels_per_unit = float(pixels_per_unit)
    updated.scale.unit = unit.strip()
    updated.scale.method = ScaleMethod.MANUAL
    updated.scale.confidence = 1.0
    updated.scale.review_status = ReviewStatus.CONFIRMED

    for wall in updated.walls:
        wall.length = wall.length_px / pixels_per_unit
    for opening in updated.openings:
        opening.width = opening.width_px / pixels_per_unit
    area_scale = pixels_per_unit * pixels_per_unit
    for room in updated.rooms:
        room.area = room.area_px / area_scale
        room.perimeter = room.perimeter_px / pixels_per_unit

    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_scale", actor, before, [updated.id],
        {
            "before": previous,
            "after": {
                "pixels_per_unit": updated.scale.pixels_per_unit,
                "unit": updated.scale.unit,
                "method": updated.scale.method.value,
                "review_status": updated.scale.review_status.value,
            },
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated

def _editable_metadata(model: TakeoffModel, object_id: str) -> ObjectMetadata:
    for collection in (
        model.nodes, model.walls, model.openings,
        model.doors, model.windows, model.rooms,
    ):
        for item in collection:
            if item.id == object_id:
                return item.metadata
    raise DomainCommandError(f"object {object_id} was not found")


def set_review_status(
    model: TakeoffModel,
    *,
    object_id: str,
    status: ReviewStatus,
    locked: bool | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Confirm, reject, or reopen one object without rerunning extraction."""
    _ensure_editable(model)
    updated = _prepare_edit(model)
    metadata = _editable_metadata(updated, object_id)
    before = updated.revision
    previous_status = metadata.review_status
    previous_locked = metadata.locked
    metadata.review_status = status
    metadata.locked = (
        status in {ReviewStatus.CONFIRMED, ReviewStatus.REJECTED}
        if locked is None else bool(locked)
    )
    metadata.revision += 1
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_review_status", actor, before, [object_id],
        {
            "before": {
                "review_status": previous_status.value,
                "locked": previous_locked,
            },
            "after": {
                "review_status": status.value,
                "locked": metadata.locked,
            },
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def _mark_manual(metadata: ObjectMetadata) -> None:
    metadata.source.kind = ObjectSourceKind.MANUAL_ADJUSTED
    metadata.source.stage = "human_edit"
    metadata.review_status = ReviewStatus.NEEDS_REVIEW
    metadata.locked = False
    metadata.revision += 1


def _manual_created_metadata(operation: str) -> ObjectMetadata:
    return ObjectMetadata(
        source=SourceEvidence(
            kind=ObjectSourceKind.MANUAL_CREATED,
            stage="human_edit",
            details={"operation": operation},
        ),
        confidence=ConfidenceBreakdown(
            overall=1.0,
            geometry_quality=1.0,
            association=1.0,
            topology_consistency=1.0,
        ),
        review_status=ReviewStatus.NEEDS_REVIEW,
    )


def _within_image(model: TakeoffModel, point: Coordinate) -> bool:
    return (
        math.isfinite(point.x)
        and math.isfinite(point.y)
        and point.x >= 0
        and point.y >= 0
        and (model.source.image_width <= 0 or point.x <= model.source.image_width)
        and (model.source.image_height <= 0 or point.y <= model.source.image_height)
    )


def _nearest_node(
    model: TakeoffModel, point: Coordinate, tolerance: float,
) -> Node | None:
    candidates = [
        (distance(node.point, point), node.id, node)
        for node in model.nodes
        if distance(node.point, point) <= tolerance
    ]
    return min(candidates, default=(0.0, "", None))[2]


def _wall_affects_room(wall: Wall, polygon: list[Coordinate]) -> bool:
    midpoint = Coordinate(
        (wall.start.x + wall.end.x) / 2.0,
        (wall.start.y + wall.end.y) / 2.0,
    )
    return point_in_polygon(midpoint, polygon) or any(
        point_on_segment(vertex, wall.start, wall.end, max(1e-6, wall.thickness_px / 2))
        for vertex in polygon
    )


def _invalidate_room_topology(room, wall_id: str) -> None:
    _mark_manual(room.metadata)
    invalidated = set(
        room.metadata.source.details.get("topology_invalidated_by_wall_ids", [])
    )
    invalidated.add(wall_id)
    room.metadata.source.details["topology_invalidated_by_wall_ids"] = sorted(invalidated)


def _resolve_room_wall_invalidation(room, wall_id: str) -> bool:
    invalidated = set(
        room.metadata.source.details.get("topology_invalidated_by_wall_ids", [])
    )
    if wall_id not in invalidated:
        return False
    invalidated.remove(wall_id)
    if invalidated:
        room.metadata.source.details["topology_invalidated_by_wall_ids"] = sorted(
            invalidated
        )
    else:
        room.metadata.source.details.pop("topology_invalidated_by_wall_ids", None)
    _mark_manual(room.metadata)
    return True


def _rebuild_wall_connectivity(model: TakeoffModel) -> None:
    nodes = {node.id: node for node in model.nodes}
    walls = {wall.id: wall for wall in model.walls}
    for node in model.nodes:
        node.connected_wall_ids = []
    for wall in model.walls:
        for node_id in (wall.start_node_id, wall.end_node_id):
            if node_id in nodes:
                nodes[node_id].connected_wall_ids.append(wall.id)
    for node in model.nodes:
        node.connected_wall_ids = sorted(set(node.connected_wall_ids))
    for wall in model.walls:
        connected: set[str] = set()
        for node_id in (wall.start_node_id, wall.end_node_id):
            if node_id in nodes:
                connected.update(nodes[node_id].connected_wall_ids)
        connected.discard(wall.id)
        wall.connected_wall_ids = sorted(item for item in connected if item in walls)


def _wall_by_id(model: TakeoffModel, wall_id: str):
    try:
        return next(wall for wall in model.walls if wall.id == wall_id)
    except StopIteration as exc:
        raise DomainCommandError(f"wall {wall_id} was not found") from exc


def _opening_by_id(model: TakeoffModel, opening_id: str) -> Opening:
    try:
        return next(item for item in model.openings if item.id == opening_id)
    except StopIteration as exc:
        raise DomainCommandError(f"opening {opening_id} was not found") from exc


def _constrained_opening_range(
    model: TakeoffModel,
    wall: Wall,
    *,
    center: Coordinate,
    width_px: float,
    projection_tolerance_px: float | None,
    exclude_opening_id: str | None = None,
) -> tuple[Coordinate, float, float, float]:
    if not _within_image(model, center):
        raise DomainCommandError(
            "opening center must be finite and within the plan image"
        )
    if not math.isfinite(width_px) or width_px <= 0:
        raise DomainCommandError("opening width must be a positive finite number")
    tolerance = (
        max(2.0, wall.thickness_px)
        if projection_tolerance_px is None else projection_tolerance_px
    )
    if not math.isfinite(tolerance) or tolerance < 0:
        raise DomainCommandError(
            "projection tolerance must be a non-negative finite number"
        )
    projected, center_offset, lateral_distance = project_point_to_segment(
        center, wall.start, wall.end,
    )
    if lateral_distance > tolerance:
        raise DomainCommandError(
            f"opening center is {lateral_distance:.3f} px from the wall centerline, "
            f"beyond tolerance {tolerance:.3f} px"
        )
    start_offset = center_offset - width_px / 2.0
    end_offset = center_offset + width_px / 2.0
    if start_offset < -1e-6 or end_offset > wall.length_px + 1e-6:
        raise DomainCommandError("opening must remain fully within its host wall")
    for other in model.openings:
        if other.wall_id != wall.id or other.id == exclude_opening_id:
            continue
        overlap = min(end_offset, other.end_offset_px) - max(
            start_offset, other.start_offset_px
        )
        if overlap > 1e-6:
            raise DomainCommandError(
                f"opening overlaps existing opening {other.id}"
            )
    return projected, max(0.0, start_offset), min(wall.length_px, end_offset), tolerance


def move_wall_endpoint(
    model: TakeoffModel,
    *,
    wall_id: str,
    endpoint: str,
    point: Coordinate,
    actor: str = "user",
) -> TakeoffModel:
    """Move one shared graph node and recompute only its dependents."""
    _ensure_editable(model)
    if endpoint not in {"start", "end"}:
        raise DomainCommandError("endpoint must be 'start' or 'end'")
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        raise DomainCommandError("endpoint coordinates must be finite")
    if point.x < 0 or point.y < 0:
        raise DomainCommandError("endpoint must remain within the plan image")
    if (
        model.source.image_width > 0 and point.x > model.source.image_width
    ) or (
        model.source.image_height > 0 and point.y > model.source.image_height
    ):
        raise DomainCommandError("endpoint must remain within the plan image")

    selected = _wall_by_id(model, wall_id)
    node_id = selected.start_node_id if endpoint == "start" else selected.end_node_id
    try:
        source_node = next(node for node in model.nodes if node.id == node_id)
    except StopIteration as exc:
        raise DomainCommandError(f"shared node {node_id} was not found") from exc

    incident_ids = set(source_node.connected_wall_ids)
    if wall_id not in incident_ids:
        raise DomainCommandError("wall endpoint is inconsistent with its graph node")
    old_point = source_node.point
    if distance(old_point, point) <= 1e-9:
        raise DomainCommandError("endpoint did not move")

    source_walls = {wall.id: wall for wall in model.walls if wall.id in incident_ids}
    opening_by_wall: dict[str, list] = {}
    for opening in model.openings:
        opening_by_wall.setdefault(opening.wall_id, []).append(opening)
    for incident_id, wall in source_walls.items():
        new_start = point if wall.start_node_id == node_id else wall.start
        new_end = point if wall.end_node_id == node_id else wall.end
        new_length = distance(new_start, new_end)
        minimum = max(1.0, wall.thickness_px * 0.25)
        if new_length < minimum:
            raise DomainCommandError(
                f"edit would collapse wall {incident_id} below {minimum:.3f} px"
            )
        furthest_opening = max(
            (opening.end_offset_px for opening in opening_by_wall.get(incident_id, [])),
            default=0.0,
        )
        if furthest_opening > new_length + 1e-3:
            raise DomainCommandError(
                f"edit would move opening beyond wall {incident_id}; "
                "move or resize the opening first"
            )

    updated = _prepare_edit(model)
    before_revision = updated.revision
    changed_ids: set[str] = {node_id}
    node = next(item for item in updated.nodes if item.id == node_id)
    node.point = point
    _mark_manual(node.metadata)

    openings = {opening.id: opening for opening in updated.openings}
    doors_by_opening: dict[str, list] = {}
    for door in updated.doors:
        doors_by_opening.setdefault(door.opening_id, []).append(door)
    windows_by_opening: dict[str, list] = {}
    for window in updated.windows:
        windows_by_opening.setdefault(window.opening_id, []).append(window)
    affected_wall_ids = set(node.connected_wall_ids)
    for wall in updated.walls:
        if wall.id not in affected_wall_ids:
            continue
        old_start, old_end = wall.start, wall.end
        if wall.start_node_id == node_id:
            wall.start = point
        if wall.end_node_id == node_id:
            wall.end = point
        wall.length_px = distance(wall.start, wall.end)
        wall.length = (
            wall.length_px / updated.scale.pixels_per_unit
            if updated.scale.pixels_per_unit else None
        )
        wall.orientation = wall_orientation(wall.start, wall.end)
        wall.polygon = wall_polygon(wall.start, wall.end, wall.thickness_px)
        _mark_manual(wall.metadata)
        changed_ids.add(wall.id)

        for opening_id in wall.opening_ids:
            opening = openings.get(opening_id)
            if opening is None:
                continue
            opening.center = point_at_offset(
                wall.start, wall.end,
                (opening.start_offset_px + opening.end_offset_px) / 2.0,
            )
            opening.orientation = wall.orientation
            _mark_manual(opening.metadata)
            changed_ids.add(opening.id)
            for door in doors_by_opening.get(opening.id, []):
                if door.hinge is not None:
                    door.hinge = transform_wall_local_point(
                        door.hinge, old_start, old_end, wall.start, wall.end,
                    )
                if door.swing_end is not None:
                    door.swing_end = transform_wall_local_point(
                        door.swing_end, old_start, old_end, wall.start, wall.end,
                    )
                door.swing_arc = [
                    transform_wall_local_point(
                        arc_point, old_start, old_end, wall.start, wall.end,
                    )
                    for arc_point in door.swing_arc
                ]
                _mark_manual(door.metadata)
                changed_ids.add(door.id)
            for window in windows_by_opening.get(opening.id, []):
                _mark_manual(window.metadata)
                changed_ids.add(window.id)

    tolerance = max(
        (wall.thickness_px for wall in updated.walls if wall.id in affected_wall_ids),
        default=1.0,
    ) * 0.25
    for room in updated.rooms:
        new_polygon = [
            point if distance(vertex, old_point) <= tolerance else vertex
            for vertex in room.polygon
        ]
        if new_polygon == room.polygon:
            continue
        room.polygon = new_polygon
        room.area_px = polygon_area(new_polygon)
        room.perimeter_px = polygon_perimeter(new_polygon)
        if updated.scale.pixels_per_unit:
            room.area = room.area_px / updated.scale.pixels_per_unit ** 2
            room.perimeter = room.perimeter_px / updated.scale.pixels_per_unit
        _mark_manual(room.metadata)
        changed_ids.add(room.id)

    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "move_wall_endpoint", actor, before_revision,
        list(changed_ids),
        {
            "wall_id": wall_id,
            "endpoint": endpoint,
            "node_id": node_id,
            "before": {"x": old_point.x, "y": old_point.y},
            "after": {"x": point.x, "y": point.y},
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def add_wall(
    model: TakeoffModel,
    *,
    start: Coordinate,
    end: Coordinate,
    thickness_px: float,
    wall_type: str = "unknown",
    snap_tolerance_px: float | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Create one constrained wall and update reciprocal graph connectivity."""
    _ensure_editable(model)
    if not _within_image(model, start) or not _within_image(model, end):
        raise DomainCommandError("wall endpoints must be finite and within the plan image")
    if not math.isfinite(thickness_px) or thickness_px <= 0:
        raise DomainCommandError("wall thickness must be a positive finite number")
    if not wall_type.strip():
        raise DomainCommandError("wall type is required")
    tolerance = (
        max(2.0, thickness_px * 0.75)
        if snap_tolerance_px is None else snap_tolerance_px
    )
    if not math.isfinite(tolerance) or tolerance < 0:
        raise DomainCommandError("snap tolerance must be a non-negative finite number")

    updated = _prepare_edit(model)
    start_node = _nearest_node(updated, start, tolerance)
    end_node = _nearest_node(updated, end, tolerance)
    start_created = start_node is None
    end_created = end_node is None
    if start_node is None:
        start_node = Node(
            id=f"node_manual_{uuid.uuid4().hex}", point=start,
            connected_wall_ids=[], metadata=_manual_created_metadata("add_wall"),
        )
        updated.nodes.append(start_node)
    if end_node is None:
        end_node = Node(
            id=f"node_manual_{uuid.uuid4().hex}", point=end,
            connected_wall_ids=[], metadata=_manual_created_metadata("add_wall"),
        )
        updated.nodes.append(end_node)
    if start_node.id == end_node.id:
        raise DomainCommandError("wall endpoints snap to the same graph node")

    snapped_start, snapped_end = start_node.point, end_node.point
    length_px = distance(snapped_start, snapped_end)
    if length_px < max(1.0, thickness_px * 0.25):
        raise DomainCommandError("wall is too short relative to its thickness")
    for existing in updated.walls:
        if {existing.start_node_id, existing.end_node_id} == {
            start_node.id, end_node.id,
        }:
            raise DomainCommandError(f"wall duplicates existing wall {existing.id}")
        if segments_intersect(
            snapped_start, snapped_end, existing.start, existing.end,
        ):
            raise DomainCommandError(
                f"wall intersects {existing.id} without a shared endpoint; "
                "split the intersected wall first"
            )

    incident_ids = {
        *start_node.connected_wall_ids,
        *end_node.connected_wall_ids,
    }
    wall_id = f"wall_manual_{uuid.uuid4().hex}"
    wall = Wall(
        id=wall_id,
        start_node_id=start_node.id,
        end_node_id=end_node.id,
        start=snapped_start,
        end=snapped_end,
        polygon=wall_polygon(snapped_start, snapped_end, thickness_px),
        thickness_px=float(thickness_px),
        wall_type=wall_type.strip(),
        orientation=wall_orientation(snapped_start, snapped_end),
        connected_wall_ids=[],
        opening_ids=[],
        length_px=length_px,
        length=(
            length_px / updated.scale.pixels_per_unit
            if updated.scale.pixels_per_unit else None
        ),
        metadata=_manual_created_metadata("add_wall"),
    )
    updated.walls.append(wall)
    changed_ids: set[str] = {wall_id, start_node.id, end_node.id, *incident_ids}
    if not start_created:
        _mark_manual(start_node.metadata)
    if not end_created:
        _mark_manual(end_node.metadata)
    for existing in updated.walls:
        if existing.id in incident_ids:
            _mark_manual(existing.metadata)
    _rebuild_wall_connectivity(updated)
    affected_room_ids: list[str] = []
    for room in updated.rooms:
        if _wall_affects_room(wall, room.polygon):
            _invalidate_room_topology(room, wall_id)
            affected_room_ids.append(room.id)
            changed_ids.add(room.id)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "add_wall", actor, before, list(changed_ids),
        {
            "wall_id": wall_id,
            "start_node_id": start_node.id,
            "end_node_id": end_node.id,
            "start_node_created": start_created,
            "end_node_created": end_created,
            "snap_tolerance_px": tolerance,
            "affected_room_ids": sorted(affected_room_ids),
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def split_wall(
    model: TakeoffModel,
    *,
    wall_id: str,
    point: Coordinate,
    projection_tolerance_px: float | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Split a wall at a projected interior point while preserving dependencies."""
    _ensure_editable(model)
    if not _within_image(model, point):
        raise DomainCommandError("split point must be finite and within the plan image")
    selected = _wall_by_id(model, wall_id)
    tolerance = (
        max(2.0, selected.thickness_px)
        if projection_tolerance_px is None else projection_tolerance_px
    )
    if not math.isfinite(tolerance) or tolerance < 0:
        raise DomainCommandError(
            "projection tolerance must be a non-negative finite number"
        )
    projected, split_offset, lateral_distance = project_point_to_segment(
        point, selected.start, selected.end,
    )
    if lateral_distance > tolerance:
        raise DomainCommandError(
            f"split point is {lateral_distance:.3f} px from the wall centerline, "
            f"beyond tolerance {tolerance:.3f} px"
        )
    endpoint_margin = max(1.0, selected.thickness_px * 0.25)
    if (
        split_offset < endpoint_margin
        or selected.length_px - split_offset < endpoint_margin
    ):
        raise DomainCommandError("split point must remain inside both wall endpoints")

    hosted_openings = [
        opening for opening in model.openings if opening.wall_id == wall_id
    ]
    spanning = [
        opening.id for opening in hosted_openings
        if opening.start_offset_px + 1e-6 < split_offset
        < opening.end_offset_px - 1e-6
    ]
    if spanning:
        raise DomainCommandError(
            "split point crosses opening(s): " + ", ".join(sorted(spanning))
            + "; move or resize them first"
        )

    updated = _prepare_edit(model)
    selected = _wall_by_id(updated, wall_id)
    old_end = selected.end
    old_end_node_id = selected.end_node_id
    split_node = _nearest_node(updated, projected, tolerance)
    reused_split_node = False
    if split_node is not None:
        _, node_offset, node_lateral = project_point_to_segment(
            split_node.point, selected.start, selected.end,
        )
        if (
            node_lateral <= 1e-3
            and endpoint_margin <= node_offset <= selected.length_px - endpoint_margin
        ):
            projected = split_node.point
            split_offset = node_offset
            reused_split_node = True
        else:
            split_node = None
    if split_node is None:
        split_node = Node(
            id=f"node_manual_{uuid.uuid4().hex}",
            point=projected,
            connected_wall_ids=[],
            metadata=_manual_created_metadata("split_wall"),
        )
        updated.nodes.append(split_node)
    spanning_after_snap = [
        opening.id for opening in hosted_openings
        if opening.start_offset_px + 1e-6 < split_offset
        < opening.end_offset_px - 1e-6
    ]
    if spanning_after_snap:
        raise DomainCommandError(
            "snapped split point crosses opening(s): "
            + ", ".join(sorted(spanning_after_snap))
            + "; move or resize them first"
        )

    existing_at_old_end = {
        wall.id for wall in updated.walls
        if wall.id != wall_id and old_end_node_id in {
            wall.start_node_id, wall.end_node_id,
        }
    }
    existing_at_split = set(split_node.connected_wall_ids)
    new_wall_id = f"wall_manual_{uuid.uuid4().hex}"
    original_start_node_id = selected.start_node_id
    original_type = selected.wall_type
    original_thickness = selected.thickness_px
    original_length = selected.length_px

    selected.end_node_id = split_node.id
    selected.end = projected
    selected.length_px = distance(selected.start, selected.end)
    selected.length = (
        selected.length_px / updated.scale.pixels_per_unit
        if updated.scale.pixels_per_unit else None
    )
    selected.orientation = wall_orientation(selected.start, selected.end)
    selected.polygon = wall_polygon(
        selected.start, selected.end, selected.thickness_px,
    )
    _mark_manual(selected.metadata)

    new_wall = Wall(
        id=new_wall_id,
        start_node_id=split_node.id,
        end_node_id=old_end_node_id,
        start=projected,
        end=old_end,
        polygon=wall_polygon(projected, old_end, original_thickness),
        thickness_px=original_thickness,
        wall_type=original_type,
        orientation=wall_orientation(projected, old_end),
        connected_wall_ids=[],
        opening_ids=[],
        length_px=distance(projected, old_end),
        length=(
            distance(projected, old_end) / updated.scale.pixels_per_unit
            if updated.scale.pixels_per_unit else None
        ),
        metadata=_manual_created_metadata("split_wall"),
    )
    new_wall.metadata.source.details["parent_wall_id"] = wall_id
    updated.walls.append(new_wall)

    selected.opening_ids = []
    doors_by_opening: dict[str, list] = {}
    for door in updated.doors:
        doors_by_opening.setdefault(door.opening_id, []).append(door)
    windows_by_opening: dict[str, list] = {}
    for window in updated.windows:
        windows_by_opening.setdefault(window.opening_id, []).append(window)
    reassigned_opening_ids: list[str] = []
    changed_ids: set[str] = {
        wall_id, new_wall_id, split_node.id, old_end_node_id,
        *existing_at_old_end, *existing_at_split,
    }
    for opening in updated.openings:
        if opening.wall_id != wall_id:
            continue
        if opening.start_offset_px >= split_offset - 1e-6:
            opening.wall_id = new_wall_id
            opening.start_offset_px -= split_offset
            opening.end_offset_px -= split_offset
            opening.center = point_at_offset(
                new_wall.start, new_wall.end,
                (opening.start_offset_px + opening.end_offset_px) / 2.0,
            )
            opening.orientation = new_wall.orientation
            new_wall.opening_ids.append(opening.id)
            _mark_manual(opening.metadata)
            reassigned_opening_ids.append(opening.id)
            changed_ids.add(opening.id)
            for dependent in [
                *doors_by_opening.get(opening.id, []),
                *windows_by_opening.get(opening.id, []),
            ]:
                _mark_manual(dependent.metadata)
                changed_ids.add(dependent.id)
        else:
            selected.opening_ids.append(opening.id)
    selected.opening_ids.sort()
    new_wall.opening_ids.sort()

    if reused_split_node:
        _mark_manual(split_node.metadata)
    for node in updated.nodes:
        if node.id == old_end_node_id:
            _mark_manual(node.metadata)
    for adjacent in updated.walls:
        if adjacent.id in existing_at_old_end | existing_at_split:
            _mark_manual(adjacent.metadata)

    affected_room_ids: list[str] = []
    for room in updated.rooms:
        if wall_id in room.boundary_wall_ids:
            expanded: list[str] = []
            for boundary_id in room.boundary_wall_ids:
                expanded.append(boundary_id)
                if boundary_id == wall_id:
                    expanded.append(new_wall_id)
            room.boundary_wall_ids = expanded
            affected_room_ids.append(room.id)
            changed_ids.add(room.id)
        invalidated = set(
            room.metadata.source.details.get("topology_invalidated_by_wall_ids", [])
        )
        if wall_id in invalidated:
            invalidated.add(new_wall_id)
            room.metadata.source.details["topology_invalidated_by_wall_ids"] = sorted(
                invalidated
            )
            _mark_manual(room.metadata)
            if room.id not in affected_room_ids:
                affected_room_ids.append(room.id)
            changed_ids.add(room.id)

    _rebuild_wall_connectivity(updated)
    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "split_wall", actor, before, list(changed_ids),
        {
            "wall_id": wall_id,
            "new_wall_id": new_wall_id,
            "split_node_id": split_node.id,
            "split_node_reused": reused_split_node,
            "requested_point": {"x": point.x, "y": point.y},
            "projected_point": {"x": projected.x, "y": projected.y},
            "split_offset_px": split_offset,
            "original_length_px": original_length,
            "original_start_node_id": original_start_node_id,
            "reassigned_opening_ids": sorted(reassigned_opening_ids),
            "affected_room_ids": sorted(affected_room_ids),
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def add_opening(
    model: TakeoffModel,
    *,
    wall_id: str,
    center: Coordinate,
    width_px: float,
    kind: OpeningKind = OpeningKind.UNKNOWN,
    projection_tolerance_px: float | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Create one logical wall-hosted opening from a constrained plan click."""
    _ensure_editable(model)
    wall = _wall_by_id(model, wall_id)
    projected, start_offset, end_offset, tolerance = _constrained_opening_range(
        model,
        wall,
        center=center,
        width_px=width_px,
        projection_tolerance_px=projection_tolerance_px,
    )

    updated = _prepare_edit(model)
    wall = _wall_by_id(updated, wall_id)
    opening_id = f"opening_manual_{uuid.uuid4().hex}"
    opening = Opening(
        id=opening_id,
        wall_id=wall_id,
        kind=kind,
        start_offset_px=start_offset,
        end_offset_px=end_offset,
        center=projected,
        width_px=float(width_px),
        width=(
            width_px / updated.scale.pixels_per_unit
            if updated.scale.pixels_per_unit else None
        ),
        orientation=wall.orientation,
        metadata=_manual_created_metadata("add_opening"),
    )
    updated.openings.append(opening)
    wall.opening_ids.append(opening_id)
    wall.opening_ids = sorted(set(wall.opening_ids))
    _mark_manual(wall.metadata)
    changed_ids: set[str] = {wall_id, opening_id}
    logical_id: str | None = None
    if kind == OpeningKind.DOOR:
        logical = Door(
            id=f"door_manual_{uuid.uuid4().hex}",
            opening_id=opening_id,
            subtype="unknown",
            swing_direction=None,
            hinge_side=None,
            hinge=None,
            swing_end=None,
            swing_arc=[],
            metadata=_manual_created_metadata("add_opening"),
        )
        updated.doors.append(logical)
        logical_id = logical.id
    elif kind == OpeningKind.WINDOW:
        logical = Window(
            id=f"window_manual_{uuid.uuid4().hex}",
            opening_id=opening_id,
            subtype="unknown",
            sill_height=None,
            metadata=_manual_created_metadata("add_opening"),
        )
        updated.windows.append(logical)
        logical_id = logical.id
    if logical_id is not None:
        changed_ids.add(logical_id)
        for room in updated.rooms:
            if wall_id not in room.boundary_wall_ids:
                continue
            target = room.door_ids if kind == OpeningKind.DOOR else room.window_ids
            target.append(logical_id)
            target[:] = sorted(set(target))
            _mark_manual(room.metadata)
            changed_ids.add(room.id)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "add_opening", actor, before, list(changed_ids),
        {
            "wall_id": wall_id,
            "opening_id": opening_id,
            "logical_object_id": logical_id,
            "kind": kind.value,
            "projected_center": {"x": projected.x, "y": projected.y},
            "start_offset_px": start_offset,
            "end_offset_px": end_offset,
            "projection_tolerance_px": tolerance,
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def update_opening_geometry(
    model: TakeoffModel,
    *,
    opening_id: str,
    center: Coordinate,
    width_px: float,
    projection_tolerance_px: float | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Move and resize an opening along its current host wall."""
    _ensure_editable(model)
    selected = _opening_by_id(model, opening_id)
    wall = _wall_by_id(model, selected.wall_id)
    projected, start_offset, end_offset, tolerance = _constrained_opening_range(
        model,
        wall,
        center=center,
        width_px=width_px,
        projection_tolerance_px=projection_tolerance_px,
        exclude_opening_id=opening_id,
    )
    if (
        distance(selected.center, projected) <= 1e-9
        and abs(selected.width_px - width_px) <= 1e-9
    ):
        raise DomainCommandError("opening geometry did not change")

    updated = _prepare_edit(model)
    opening = _opening_by_id(updated, opening_id)
    wall = _wall_by_id(updated, opening.wall_id)
    previous = {
        "center": {"x": opening.center.x, "y": opening.center.y},
        "width_px": opening.width_px,
        "start_offset_px": opening.start_offset_px,
        "end_offset_px": opening.end_offset_px,
    }
    dx = projected.x - opening.center.x
    dy = projected.y - opening.center.y
    width_ratio = width_px / opening.width_px
    opening.center = projected
    opening.start_offset_px = start_offset
    opening.end_offset_px = end_offset
    opening.width_px = float(width_px)
    opening.width = (
        width_px / updated.scale.pixels_per_unit
        if updated.scale.pixels_per_unit else None
    )
    opening.orientation = wall.orientation
    _mark_manual(opening.metadata)
    _mark_manual(wall.metadata)
    changed_ids: set[str] = {opening_id, wall.id}
    for door in updated.doors:
        if door.opening_id != opening_id:
            continue
        old_hinge = door.hinge
        if door.hinge is not None:
            door.hinge = Coordinate(door.hinge.x + dx, door.hinge.y + dy)
        if door.swing_end is not None:
            if old_hinge is not None and door.hinge is not None:
                door.swing_end = Coordinate(
                    door.hinge.x + (door.swing_end.x - old_hinge.x) * width_ratio,
                    door.hinge.y + (door.swing_end.y - old_hinge.y) * width_ratio,
                )
            else:
                door.swing_end = Coordinate(
                    door.swing_end.x + dx, door.swing_end.y + dy,
                )
        if old_hinge is not None and door.hinge is not None:
            door.swing_arc = [
                Coordinate(
                    door.hinge.x + (point.x - old_hinge.x) * width_ratio,
                    door.hinge.y + (point.y - old_hinge.y) * width_ratio,
                )
                for point in door.swing_arc
            ]
        else:
            door.swing_arc = [
                Coordinate(point.x + dx, point.y + dy) for point in door.swing_arc
            ]
        _mark_manual(door.metadata)
        changed_ids.add(door.id)
    for window in updated.windows:
        if window.opening_id == opening_id:
            _mark_manual(window.metadata)
            changed_ids.add(window.id)
    for room in updated.rooms:
        if wall.id in room.boundary_wall_ids:
            _mark_manual(room.metadata)
            changed_ids.add(room.id)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "update_opening_geometry", actor, before, list(changed_ids),
        {
            "opening_id": opening_id,
            "wall_id": wall.id,
            "before": previous,
            "after": {
                "center": {"x": projected.x, "y": projected.y},
                "width_px": width_px,
                "start_offset_px": start_offset,
                "end_offset_px": end_offset,
            },
            "projection_tolerance_px": tolerance,
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def set_opening_kind(
    model: TakeoffModel,
    *,
    opening_id: str,
    kind: OpeningKind,
    actor: str = "user",
) -> TakeoffModel:
    """Reclassify one physical opening while preserving its stable ID."""
    _ensure_editable(model)
    selected = _opening_by_id(model, opening_id)
    if selected.kind == kind:
        raise DomainCommandError(f"opening is already classified as {kind.value}")

    updated = _prepare_edit(model)
    opening = _opening_by_id(updated, opening_id)
    wall = _wall_by_id(updated, opening.wall_id)
    removed_door_ids = {
        door.id for door in updated.doors if door.opening_id == opening_id
    }
    removed_window_ids = {
        window.id for window in updated.windows if window.opening_id == opening_id
    }
    updated.doors = [
        door for door in updated.doors if door.id not in removed_door_ids
    ]
    updated.windows = [
        window for window in updated.windows if window.id not in removed_window_ids
    ]
    previous_kind = opening.kind
    opening.kind = kind
    _mark_manual(opening.metadata)
    _mark_manual(wall.metadata)
    logical_id: str | None = None
    if kind == OpeningKind.DOOR:
        door = Door(
            id=f"door_manual_{uuid.uuid4().hex}",
            opening_id=opening_id,
            subtype="unknown",
            swing_direction=None,
            hinge_side=None,
            hinge=None,
            swing_end=None,
            swing_arc=[],
            metadata=_manual_created_metadata("set_opening_kind"),
        )
        updated.doors.append(door)
        logical_id = door.id
    elif kind == OpeningKind.WINDOW:
        window = Window(
            id=f"window_manual_{uuid.uuid4().hex}",
            opening_id=opening_id,
            subtype="unknown",
            sill_height=None,
            metadata=_manual_created_metadata("set_opening_kind"),
        )
        updated.windows.append(window)
        logical_id = window.id

    changed_ids: set[str] = {
        opening_id, wall.id, *removed_door_ids, *removed_window_ids,
    }
    if logical_id is not None:
        changed_ids.add(logical_id)
    old_logical_ids = removed_door_ids | removed_window_ids
    for room in updated.rooms:
        had_relationship = bool(
            old_logical_ids.intersection(room.door_ids)
            or old_logical_ids.intersection(room.window_ids)
        )
        room.door_ids = [
            item for item in room.door_ids if item not in old_logical_ids
        ]
        room.window_ids = [
            item for item in room.window_ids if item not in old_logical_ids
        ]
        if logical_id is not None and (
            had_relationship or wall.id in room.boundary_wall_ids
        ):
            target = room.door_ids if kind == OpeningKind.DOOR else room.window_ids
            target.append(logical_id)
            target[:] = sorted(set(target))
        if had_relationship or wall.id in room.boundary_wall_ids:
            _mark_manual(room.metadata)
            changed_ids.add(room.id)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_opening_kind", actor, before, list(changed_ids),
        {
            "opening_id": opening_id,
            "wall_id": wall.id,
            "before": previous_kind.value,
            "after": kind.value,
            "removed_door_ids": sorted(removed_door_ids),
            "removed_window_ids": sorted(removed_window_ids),
            "new_logical_object_id": logical_id,
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def delete_opening(
    model: TakeoffModel,
    *,
    opening_id: str,
    cascade: bool = False,
    actor: str = "user",
) -> TakeoffModel:
    """Delete one physical opening with explicit logical dependency consent."""
    _ensure_editable(model)
    selected = _opening_by_id(model, opening_id)
    door_ids = {
        door.id for door in model.doors if door.opening_id == opening_id
    }
    window_ids = {
        window.id for window in model.windows if window.opening_id == opening_id
    }
    dependents = sorted(door_ids | window_ids)
    if dependents and not cascade:
        raise DomainCommandError(
            "opening has dependent objects: " + ", ".join(dependents)
            + "; retry with cascade=true"
        )

    updated = _prepare_edit(model)
    opening = _opening_by_id(updated, opening_id)
    wall = _wall_by_id(updated, opening.wall_id)
    if cascade:
        updated.doors = [door for door in updated.doors if door.id not in door_ids]
        updated.windows = [
            window for window in updated.windows if window.id not in window_ids
        ]
    updated.openings = [
        item for item in updated.openings if item.id != opening_id
    ]
    wall.opening_ids = [item for item in wall.opening_ids if item != opening_id]
    _mark_manual(wall.metadata)
    changed_ids: set[str] = {opening_id, wall.id, *door_ids, *window_ids}
    logical_ids = door_ids | window_ids
    for room in updated.rooms:
        had_relationship = bool(
            logical_ids.intersection(room.door_ids)
            or logical_ids.intersection(room.window_ids)
        )
        room.door_ids = [item for item in room.door_ids if item not in logical_ids]
        room.window_ids = [
            item for item in room.window_ids if item not in logical_ids
        ]
        if had_relationship or wall.id in room.boundary_wall_ids:
            _mark_manual(room.metadata)
            changed_ids.add(room.id)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "delete_opening", actor, before, list(changed_ids),
        {
            "opening_id": opening_id,
            "wall_id": wall.id,
            "kind": opening.kind.value,
            "cascade": cascade,
            "deleted_door_ids": sorted(door_ids),
            "deleted_window_ids": sorted(window_ids),
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def delete_wall(
    model: TakeoffModel,
    *,
    wall_id: str,
    cascade: bool = False,
    actor: str = "user",
) -> TakeoffModel:
    """Delete a wall, requiring explicit cascade for logical dependents."""
    _ensure_editable(model)
    selected = _wall_by_id(model, wall_id)
    opening_ids = {
        opening.id for opening in model.openings if opening.wall_id == wall_id
    } | set(selected.opening_ids)
    door_ids = {
        door.id for door in model.doors if door.opening_id in opening_ids
    }
    window_ids = {
        window.id for window in model.windows if window.opening_id in opening_ids
    }
    boundary_room_ids = {
        room.id for room in model.rooms if wall_id in room.boundary_wall_ids
    }
    dependents = sorted(opening_ids | door_ids | window_ids | boundary_room_ids)
    if dependents and not cascade:
        raise DomainCommandError(
            "wall has dependent objects: " + ", ".join(dependents)
            + "; retry with cascade=true"
        )

    updated = _prepare_edit(model)
    selected = _wall_by_id(updated, wall_id)
    endpoint_ids = {selected.start_node_id, selected.end_node_id}
    adjacent_ids = {
        wall.id for wall in updated.walls
        if wall.id != wall_id
        and endpoint_ids.intersection({wall.start_node_id, wall.end_node_id})
    }
    changed_ids: set[str] = {
        wall_id, *endpoint_ids, *adjacent_ids,
        *opening_ids, *door_ids, *window_ids,
    }
    if cascade:
        updated.openings = [
            opening for opening in updated.openings if opening.id not in opening_ids
        ]
        updated.doors = [door for door in updated.doors if door.id not in door_ids]
        updated.windows = [
            window for window in updated.windows if window.id not in window_ids
        ]
    updated.walls = [wall for wall in updated.walls if wall.id != wall_id]

    for adjacent in updated.walls:
        if adjacent.id in adjacent_ids:
            _mark_manual(adjacent.metadata)
    for room in updated.rooms:
        formally_affected = wall_id in room.boundary_wall_ids
        spatially_affected = _wall_affects_room(selected, room.polygon)
        if not formally_affected and not spatially_affected:
            continue
        room.boundary_wall_ids = [
            item for item in room.boundary_wall_ids if item != wall_id
        ]
        if cascade:
            room.door_ids = [item for item in room.door_ids if item not in door_ids]
            room.window_ids = [
                item for item in room.window_ids if item not in window_ids
            ]
        if not _resolve_room_wall_invalidation(room, wall_id):
            _invalidate_room_topology(room, wall_id)
        changed_ids.add(room.id)

    _rebuild_wall_connectivity(updated)
    orphan_node_ids = {
        node.id for node in updated.nodes if not node.connected_wall_ids
    }
    changed_ids.update(orphan_node_ids)
    updated.nodes = [
        node for node in updated.nodes if node.id not in orphan_node_ids
    ]
    for node in updated.nodes:
        if node.id in endpoint_ids:
            _mark_manual(node.metadata)

    before = updated.revision
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "delete_wall", actor, before, list(changed_ids),
        {
            "wall_id": wall_id,
            "cascade": cascade,
            "deleted_opening_ids": sorted(opening_ids),
            "deleted_door_ids": sorted(door_ids),
            "deleted_window_ids": sorted(window_ids),
            "orphan_node_ids": sorted(orphan_node_ids),
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def set_approval_status(
    model: TakeoffModel,
    *,
    status: ApprovalStatus,
    actor: str = "user",
) -> TakeoffModel:
    """Freeze a verified revision or explicitly reopen it for further edits."""
    if status == model.approval_status:
        raise DomainCommandError(f"model is already {status.value}")
    if (
        model.approval_status == ApprovalStatus.APPROVED
        and status != ApprovalStatus.IN_REVIEW
    ):
        raise DomainCommandError(
            "approved models must be reopened to in_review before other transitions"
        )

    updated = _prepare_edit(model)
    before = updated.revision
    if status == ApprovalStatus.APPROVED:
        updated.validation_issues = validate_model(updated)
        verified = calculate_quantities(updated, QuantityBasis.VERIFIED)
        if not verified.complete:
            reasons = "; ".join(verified.warnings) or "review is incomplete"
            raise DomainCommandError(
                f"model cannot be approved: {reasons}"
            )

    previous = updated.approval_status
    updated.approval_status = status
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_approval_status", actor, before, [updated.id],
        {"before": previous.value, "after": status.value},
    ))
    updated.validation_issues = validate_model(updated)
    return updated


def _changed_object_ids(first: TakeoffModel, second: TakeoffModel) -> list[str]:
    changed: set[str] = set()
    if first.scale != second.scale or first.approval_status != second.approval_status:
        changed.add(first.id)
    for first_collection, second_collection in zip(
        (first.nodes, first.walls, first.openings, first.doors, first.windows, first.rooms),
        (second.nodes, second.walls, second.openings, second.doors, second.windows, second.rooms),
    ):
        first_by_id = {item.id: item for item in first_collection}
        second_by_id = {item.id: item for item in second_collection}
        for object_id in first_by_id.keys() | second_by_id.keys():
            if first_by_id.get(object_id) != second_by_id.get(object_id):
                changed.add(object_id)
    return sorted(changed)


def _restore_snapshot(
    model: TakeoffModel,
    snapshot: TakeoffModel,
    *,
    action: str,
    actor: str,
) -> TakeoffModel:
    _ensure_editable(model)
    if snapshot.id != model.id:
        raise DomainCommandError("history snapshot belongs to a different model")
    if action == "undo":
        if not model.undo_revision_stack:
            raise DomainCommandError("nothing to undo")
        expected = model.undo_revision_stack[-1]
        if snapshot.revision != expected:
            raise DomainCommandError(
                f"undo requires revision {expected}, got {snapshot.revision}"
            )
        undo_stack = list(model.undo_revision_stack[:-1])
        redo_stack = [*model.redo_revision_stack, model.revision]
    elif action == "redo":
        if not model.redo_revision_stack:
            raise DomainCommandError("nothing to redo")
        expected = model.redo_revision_stack[-1]
        if snapshot.revision != expected:
            raise DomainCommandError(
                f"redo requires revision {expected}, got {snapshot.revision}"
            )
        undo_stack = [*model.undo_revision_stack, model.revision]
        redo_stack = list(model.redo_revision_stack[:-1])
    else:
        raise DomainCommandError("history action must be undo or redo")

    restored = copy.deepcopy(snapshot)
    changed_ids = _changed_object_ids(model, restored)
    before = model.revision
    restored.revision = before + 1
    restored.edit_history = list(model.edit_history)
    restored.undo_revision_stack = undo_stack
    restored.redo_revision_stack = redo_stack
    restored.validation_issues = validate_model(restored)
    restored.edit_history.append(_event(
        restored, action, actor, before, changed_ids or [model.id],
        {
            "restored_snapshot_revision": snapshot.revision,
            "abandoned_revision": model.revision,
        },
    ))
    return restored


def undo_last_edit(
    model: TakeoffModel,
    snapshot: TakeoffModel,
    *,
    actor: str = "user",
) -> TakeoffModel:
    return _restore_snapshot(model, snapshot, action="undo", actor=actor)


def redo_last_edit(
    model: TakeoffModel,
    snapshot: TakeoffModel,
    *,
    actor: str = "user",
) -> TakeoffModel:
    return _restore_snapshot(model, snapshot, action="redo", actor=actor)
