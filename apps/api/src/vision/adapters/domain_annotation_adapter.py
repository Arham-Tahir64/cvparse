"""Annotation documents and SVG overlays generated from reviewed domain state."""
from __future__ import annotations

from html import escape
from typing import Any

from vision.domain.geometry import point_at_offset
from vision.domain.models import ReviewStatus, TakeoffModel


def _visible(item) -> bool:
    return item.metadata.review_status != ReviewStatus.REJECTED


def _metadata(item) -> dict[str, Any]:
    return {
        "confidence": item.metadata.confidence.overall,
        "review_state": item.metadata.review_status.value,
        "locked": item.metadata.locked,
        "object_revision": item.metadata.revision,
        "source": item.metadata.source.kind.value,
    }


def to_model_annotation_document(model: TakeoffModel) -> dict[str, Any]:
    """Export only current, non-rejected domain objects for review/rendering."""
    elements: list[dict[str, Any]] = []
    walls = {wall.id: wall for wall in model.walls if _visible(wall)}
    openings = {
        opening.id: opening for opening in model.openings
        if _visible(opening) and opening.wall_id in walls
    }

    for wall in walls.values():
        elements.append({
            "type": "wall",
            "id": wall.id,
            "geometry": {
                "kind": "wall",
                "centerline": {
                    "x1": wall.start.x, "y1": wall.start.y,
                    "x2": wall.end.x, "y2": wall.end.y,
                },
                "polygon": [{"x": point.x, "y": point.y} for point in wall.polygon],
                "thickness_px": wall.thickness_px,
                "length_px": wall.length_px,
                "length": wall.length,
            },
            "relations": {
                "start_node_id": wall.start_node_id,
                "end_node_id": wall.end_node_id,
                "connected_wall_ids": list(wall.connected_wall_ids),
                "opening_ids": [
                    item for item in wall.opening_ids if item in openings
                ],
            },
            **_metadata(wall),
        })

    for room in model.rooms:
        if not _visible(room):
            continue
        elements.append({
            "type": "room",
            "id": room.id,
            "geometry": {
                "kind": "polygon",
                "polygon": [{"x": point.x, "y": point.y} for point in room.polygon],
            },
            "label": room.label,
            "area_px": room.area_px,
            "area": room.area,
            "perimeter_px": room.perimeter_px,
            "perimeter": room.perimeter,
            "relations": {
                "boundary_wall_ids": list(room.boundary_wall_ids),
                "door_ids": list(room.door_ids),
                "window_ids": list(room.window_ids),
                "neighboring_room_ids": list(room.neighboring_room_ids),
            },
            **_metadata(room),
        })

    for door in model.doors:
        opening = openings.get(door.opening_id)
        if not _visible(door) or opening is None:
            continue
        elements.append({
            "type": "door",
            "id": door.id,
            "geometry": {
                "kind": "opening_swing",
                "opening_center": {"x": opening.center.x, "y": opening.center.y},
                "width_px": opening.width_px,
                "width": opening.width,
                "hinge": (
                    {"x": door.hinge.x, "y": door.hinge.y}
                    if door.hinge is not None else None
                ),
                "leaf_end": (
                    {"x": door.swing_end.x, "y": door.swing_end.y}
                    if door.swing_end is not None else None
                ),
                "arc": [{"x": point.x, "y": point.y} for point in door.swing_arc],
                "swing_direction": door.swing_direction,
                "hinge_side": door.hinge_side,
                "subtype": door.subtype,
            },
            "relations": {
                "opening_id": opening.id,
                "wall_id": opening.wall_id,
            },
            **_metadata(door),
        })

    for window in model.windows:
        opening = openings.get(window.opening_id)
        if not _visible(window) or opening is None:
            continue
        elements.append({
            "type": "window",
            "id": window.id,
            "geometry": {
                "kind": "opening",
                "center": {"x": opening.center.x, "y": opening.center.y},
                "width_px": opening.width_px,
                "width": opening.width,
                "orientation": opening.orientation,
                "subtype": window.subtype,
            },
            "relations": {
                "opening_id": opening.id,
                "wall_id": opening.wall_id,
            },
            **_metadata(window),
        })

    classified_openings = {
        door.opening_id for door in model.doors if _visible(door)
    } | {
        window.opening_id for window in model.windows if _visible(window)
    }
    for opening in openings.values():
        if opening.id in classified_openings:
            continue
        elements.append({
            "type": "opening",
            "id": opening.id,
            "geometry": {
                "kind": "opening",
                "center": {"x": opening.center.x, "y": opening.center.y},
                "width_px": opening.width_px,
                "width": opening.width,
                "orientation": opening.orientation,
                "opening_type": opening.kind.value,
            },
            "relations": {"wall_id": opening.wall_id},
            **_metadata(opening),
        })

    return {
        "schema_version": model.schema_version,
        "model_id": model.id,
        "model_revision": model.revision,
        "image": {
            "width": model.source.image_width,
            "height": model.source.image_height,
            "dpi": model.source.dpi,
            "page_number": model.source.page_number,
        },
        "elements": elements,
    }


def _number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _points(points) -> str:
    return " ".join(f"{_number(point.x)},{_number(point.y)}" for point in points)


def _attrs(item) -> str:
    metadata = item.metadata
    values = {
        "data-id": item.id,
        "data-review-state": metadata.review_status.value,
        "data-source": metadata.source.kind.value,
        "data-object-revision": str(metadata.revision),
    }
    return " ".join(
        f'{name}="{escape(value, quote=True)}"' for name, value in values.items()
    )


def to_model_svg(model: TakeoffModel) -> str:
    """Render a deterministic transparent SVG overlay from authoritative state."""
    width = max(1, model.source.image_width)
    height = max(1, model.source.image_height)
    walls = {wall.id: wall for wall in model.walls if _visible(wall)}
    openings = {
        opening.id: opening for opening in model.openings
        if _visible(opening) and opening.wall_id in walls
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-model-id="{escape(model.id, quote=True)}" '
        f'data-model-revision="{model.revision}">',
        "<style>.needs-review{stroke-dasharray:6 4}.conflicting{stroke:#d627b0!important;stroke-width:3}</style>",
    ]

    for room in model.rooms:
        if not _visible(room) or len(room.polygon) < 3:
            continue
        state = escape(room.metadata.review_status.value, quote=True)
        parts.append(
            f'<polygon {_attrs(room)} class="room {state.replace("_", "-")}" '
            f'points="{_points(room.polygon)}" fill="#aec7e8" fill-opacity="0.25" '
            f'stroke="#7f7f7f" stroke-width="1"/>'
        )
        if room.label:
            cx = sum(point.x for point in room.polygon) / len(room.polygon)
            cy = sum(point.y for point in room.polygon) / len(room.polygon)
            parts.append(
                f'<text x="{_number(cx)}" y="{_number(cy)}" text-anchor="middle" '
                f'font-size="12" fill="#1a1a1a">{escape(room.label)}</text>'
            )

    for wall in walls.values():
        state = wall.metadata.review_status.value.replace("_", "-")
        color = "#ff7f0e" if wall.metadata.confidence.overall < 0.6 else "#d62728"
        parts.append(
            f'<polygon {_attrs(wall)} class="wall {state}" points="{_points(wall.polygon)}" '
            f'fill="{color}" fill-opacity="0.85" stroke="{color}" stroke-width="0.2"/>'
        )

    for opening in openings.values():
        wall = walls[opening.wall_id]
        start = point_at_offset(wall.start, wall.end, opening.start_offset_px)
        end = point_at_offset(wall.start, wall.end, opening.end_offset_px)
        color = "#2ca02c" if opening.kind.value == "door" else "#1f77b4"
        parts.append(
            f'<line {_attrs(opening)} class="opening" x1="{_number(start.x)}" '
            f'y1="{_number(start.y)}" x2="{_number(end.x)}" y2="{_number(end.y)}" '
            f'stroke="{color}" stroke-width="{_number(max(2.0, wall.thickness_px))}" '
            f'stroke-opacity="0.72"/>'
        )

    for door in model.doors:
        if not _visible(door) or door.opening_id not in openings:
            continue
        if door.hinge is not None and door.swing_end is not None:
            parts.append(
                f'<line {_attrs(door)} class="door" x1="{_number(door.hinge.x)}" '
                f'y1="{_number(door.hinge.y)}" x2="{_number(door.swing_end.x)}" '
                f'y2="{_number(door.swing_end.y)}" stroke="#2ca02c" stroke-width="2"/>'
            )
        if len(door.swing_arc) >= 2:
            parts.append(
                f'<polyline class="door-arc" points="{_points(door.swing_arc)}" '
                f'fill="none" stroke="#2ca02c" stroke-width="1.5"/>'
            )

    parts.append("</svg>")
    return "".join(parts)
