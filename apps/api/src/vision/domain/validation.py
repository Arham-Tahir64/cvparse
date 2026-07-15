"""Structural validation for editable takeoff models."""
from __future__ import annotations

import hashlib
import math

from .models import (
    IssueSeverity,
    OpeningKind,
    ReviewStatus,
    TakeoffModel,
    ValidationIssue,
)


def _issue(
    model: TakeoffModel,
    code: str,
    severity: IssueSeverity,
    message: str,
    affected: list[str],
    uncertainty: float,
    structural_impact: float,
    cost_impact: float,
) -> ValidationIssue:
    payload = "|".join([model.id, code, *sorted(affected)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return ValidationIssue(
        id=f"issue_{digest}", code=code, severity=severity, message=message,
        affected_object_ids=sorted(affected), uncertainty=uncertainty,
        structural_impact=structural_impact, cost_impact=cost_impact,
        priority=round(uncertainty * structural_impact * max(cost_impact, 0.1), 6),
    )


def validate_model(model: TakeoffModel) -> list[ValidationIssue]:
    """Return deterministic issues without silently repairing model geometry."""
    issues: list[ValidationIssue] = []
    if (
        model.scale.pixels_per_unit is None
        or model.scale.pixels_per_unit <= 0
        or model.scale.review_status != ReviewStatus.CONFIRMED
    ):
        issues.append(_issue(
            model, "scale.unconfirmed", IssueSeverity.ERROR,
            "Drawing scale must be confirmed before quantities are authoritative.",
            [model.id], 1.0, 1.0, 1.0,
        ))

    nodes = {node.id: node for node in model.nodes}
    walls = {wall.id: wall for wall in model.walls}
    openings = {opening.id: opening for opening in model.openings}

    for wall in model.walls:
        missing = [
            node_id for node_id in (wall.start_node_id, wall.end_node_id)
            if node_id not in nodes
        ]
        if missing:
            issues.append(_issue(
                model, "wall.dangling_node", IssueSeverity.ERROR,
                "Wall references a missing graph node.", [wall.id, *missing],
                1.0, 1.0, 0.8,
            ))
        if wall.length_px <= 0 or wall.thickness_px <= 0:
            issues.append(_issue(
                model, "wall.invalid_geometry", IssueSeverity.ERROR,
                "Wall length and thickness must be positive.", [wall.id],
                1.0, 1.0, 0.8,
            ))
        start_node = nodes.get(wall.start_node_id)
        end_node = nodes.get(wall.end_node_id)
        if (
            start_node is not None
            and math.hypot(
                wall.start.x - start_node.point.x,
                wall.start.y - start_node.point.y,
            ) > 1e-3
        ) or (
            end_node is not None
            and math.hypot(
                wall.end.x - end_node.point.x,
                wall.end.y - end_node.point.y,
            ) > 1e-3
        ):
            issues.append(_issue(
                model, "wall.node_geometry_mismatch", IssueSeverity.ERROR,
                "Wall endpoint coordinates do not match their graph nodes.",
                [wall.id, wall.start_node_id, wall.end_node_id],
                1.0, 1.0, 0.8,
            ))

    for node in model.nodes:
        inconsistent = [
            wall_id for wall_id in node.connected_wall_ids
            if wall_id not in walls or node.id not in {
                walls[wall_id].start_node_id, walls[wall_id].end_node_id,
            }
        ]
        if inconsistent:
            issues.append(_issue(
                model, "node.connectivity_mismatch", IssueSeverity.ERROR,
                "Node connectivity does not match wall endpoint references.",
                [node.id, *inconsistent], 1.0, 1.0, 0.8,
            ))

    ranges_by_wall: dict[str, list] = {}
    for opening in model.openings:
        wall = walls.get(opening.wall_id)
        if wall is None:
            issues.append(_issue(
                model, "opening.missing_wall", IssueSeverity.ERROR,
                "Opening is not attached to an existing wall.",
                [opening.id, opening.wall_id], 1.0, 1.0, 0.8,
            ))
            continue
        if (
            opening.width_px <= 0
            or opening.end_offset_px <= opening.start_offset_px
            or opening.start_offset_px < 0
            or opening.end_offset_px > wall.length_px + 1e-3
        ):
            issues.append(_issue(
                model, "opening.invalid_range", IssueSeverity.ERROR,
                "Opening range must be positive and contained by its host wall.",
                [opening.id, wall.id], 1.0, 0.9, 0.7,
            ))
        ranges_by_wall.setdefault(wall.id, []).append(opening)

    for wall_id, candidates in ranges_by_wall.items():
        ordered = sorted(candidates, key=lambda item: (item.start_offset_px, item.id))
        for first, second in zip(ordered, ordered[1:]):
            overlap = min(first.end_offset_px, second.end_offset_px) - max(
                first.start_offset_px, second.start_offset_px
            )
            if overlap > 0.5 * min(first.width_px, second.width_px):
                issues.append(_issue(
                    model, "opening.duplicate_overlap", IssueSeverity.WARNING,
                    "Two logical openings substantially overlap on one wall.",
                    [wall_id, first.id, second.id], 0.9, 0.8, 0.6,
                ))

    seen_door_openings: set[str] = set()
    for door in model.doors:
        opening = openings.get(door.opening_id)
        if opening is None or opening.kind != OpeningKind.DOOR:
            issues.append(_issue(
                model, "door.invalid_opening", IssueSeverity.ERROR,
                "Door must reference one existing door opening.",
                [door.id, door.opening_id], 1.0, 0.9, 0.6,
            ))
        if door.opening_id in seen_door_openings:
            issues.append(_issue(
                model, "door.duplicate", IssueSeverity.WARNING,
                "More than one door references the same physical opening.",
                [door.id, door.opening_id], 0.9, 0.7, 0.5,
            ))
        seen_door_openings.add(door.opening_id)

    seen_window_openings: set[str] = set()
    for window in model.windows:
        opening = openings.get(window.opening_id)
        if opening is None or opening.kind != OpeningKind.WINDOW:
            issues.append(_issue(
                model, "window.invalid_opening", IssueSeverity.ERROR,
                "Window must reference one existing window opening.",
                [window.id, window.opening_id], 1.0, 0.8, 0.6,
            ))
        if window.opening_id in seen_window_openings:
            issues.append(_issue(
                model, "window.duplicate", IssueSeverity.WARNING,
                "More than one window references the same physical opening.",
                [window.id, window.opening_id], 0.9, 0.6, 0.5,
            ))
        seen_window_openings.add(window.opening_id)

    for room in model.rooms:
        if len(room.polygon) < 3 or room.area_px <= 0:
            issues.append(_issue(
                model, "room.invalid_polygon", IssueSeverity.ERROR,
                "Room must have a positive-area polygon.", [room.id],
                1.0, 1.0, 0.9,
            ))
        invalidated_by = room.metadata.source.details.get(
            "topology_invalidated_by_wall_ids", []
        )
        if invalidated_by:
            issues.append(_issue(
                model, "room.topology_stale", IssueSeverity.ERROR,
                "Room topology must be recomputed after a structural wall edit.",
                [room.id, *invalidated_by], 1.0, 1.0, 0.9,
            ))

    return sorted(issues, key=lambda item: (-item.priority, item.code, item.id))
