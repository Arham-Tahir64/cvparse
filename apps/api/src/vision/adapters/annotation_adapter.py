"""Annotation adapter: CVTakeoffResult -> frontend annotation document.

Separate from serialize.py on purpose: frontend format changes touch only
this file, never the pipeline output schema.
"""
from __future__ import annotations

from typing import Any

from ..cv.models import CVTakeoffResult


def to_annotation_document(result: CVTakeoffResult) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []

    for wall in result.walls:
        cl = wall.centerline
        elements.append({
            "type": "wall",
            "id": wall.id,
            "geometry": {
                "kind": "segment",
                "x1": cl.start.x,
                "y1": cl.start.y,
                "x2": cl.end.x,
                "y2": cl.end.y,
                "thickness_px": wall.visual_thickness,
            },
            "relations": {
                "source_wall_ids": list(wall.source_ids),
                "merge_kind": wall.merge_kind,
                "fit_support_ratio": wall.fit_support_ratio,
                "merge_confidence": wall.merge_confidence,
            },
            "review_state": "pending",
        })

    for room in result.rooms:
        elements.append({
            "type": "room",
            "id": room.id,
            "polygon": [{"x": p.x, "y": p.y} for p in room.polygon],
            "label": room.label,
            "area_px": room.area,
        })

    for door in result.doors:
        elements.append({
            "type": "door",
            "id": door.id,
            "geometry": {
                "kind": "swing",
                "hinge": {"x": door.position.x, "y": door.position.y},
                "leaf_end": {"x": door.swing_end.x, "y": door.swing_end.y},
                "arc": [{"x": p.x, "y": p.y} for p in door.swing_arc],
                "radius_px": door.radius,
                "swing_direction": door.swing_direction,
            },
            "relations": {
                "wall_id": door.wall_id,
                "opens_into_room_id": door.opens_into_room_id,
            },
            "confidence": door.confidence,
            "review_state": "pending",
        })

    for window in result.windows:
        elements.append({
            "type": "window",
            "id": window.id,
            "geometry": {
                "kind": "opening",
                "center": {"x": window.position.x, "y": window.position.y},
                "width_px": window.width,
            },
            "relations": {"wall_id": window.wall_id},
            "review_state": "pending",
        })

    return {"elements": elements}
