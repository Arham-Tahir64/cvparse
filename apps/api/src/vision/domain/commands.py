"""Revisioned human correction commands and targeted recomputation."""
from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone

from .geometry import (
    distance,
    point_at_offset,
    polygon_area,
    polygon_perimeter,
    transform_wall_local_point,
    wall_orientation,
    wall_polygon,
)
from .models import (
    ApprovalStatus,
    Coordinate,
    EditEvent,
    ObjectMetadata,
    ObjectSourceKind,
    ReviewStatus,
    ScaleMethod,
    TakeoffModel,
)
from .validation import validate_model


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

    updated = copy.deepcopy(model)
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
    updated = copy.deepcopy(model)
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


def _wall_by_id(model: TakeoffModel, wall_id: str):
    try:
        return next(wall for wall in model.walls if wall.id == wall_id)
    except StopIteration as exc:
        raise DomainCommandError(f"wall {wall_id} was not found") from exc


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

    updated = copy.deepcopy(model)
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
