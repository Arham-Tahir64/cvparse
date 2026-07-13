"""Module 11 - Output Serialization.

Converts CVTakeoffResult to the canonical API JSON (schema 1.0.0) and an
optional SVG debug overlay. Pure formatting; no transformations.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Any, Optional

import cv2
import numpy as np

from .models import CVTakeoffResult, Point

logger = logging.getLogger("flowbuildr.cv.serialize")

MODULE = "11_serialize"

SCHEMA_VERSION = "1.0.0"

_LOW_CONFIDENCE = 0.6  # walls below this render orange in the SVG


def _num(value) -> Optional[float]:
    """Round to 3 decimals; NaN/Inf become None with a debug log."""
    if value is None:
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        logger.debug("non-finite coordinate serialized as null: %r", value)
        return None
    return round(f, 3)


def _point(p: Point) -> dict:
    return {"x": _num(p.x), "y": _num(p.y)}


def to_json_dict(result: CVTakeoffResult) -> dict[str, Any]:
    meta = result.metadata
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "source_path": meta.source_path if meta else None,
            "image_width": meta.image_width if meta else 0,
            "image_height": meta.image_height if meta else 0,
            "dpi": meta.dpi if meta else 0,
            "page_number": meta.page_number if meta else 0,
            "wall_count": meta.wall_count if meta else 0,
            "room_count": meta.room_count if meta else 0,
        },
        "walls": [
            {
                "id": w.id,
                "orientation": w.orientation,
                "start": _point(w.centerline.start),
                "end": _point(w.centerline.end),
                "thickness": _num(w.thickness),
                "visual_thickness": _num(w.visual_thickness),
                "wall_type": w.wall_type,
                "merge_kind": w.merge_kind,
                "fit_support_ratio": _num(w.fit_support_ratio),
                "merge_confidence": _num(w.merge_confidence),
                "source_ids": list(w.source_ids),
                "length_px": _num(w.length_px),
                "length_ft": _num(w.length_ft),
            }
            for w in result.walls
        ],
        "gaps": [
            {
                "id": g.id,
                "wall_id": g.wall_id,
                "orientation": g.orientation,
                "kind": g.kind,
                "center": _point(g.center),
                "width_px": _num(g.width_px),
                "bbox": [_num(v) for v in g.bbox],
                "wall_break_score": _num(g.wall_break_score),
                "opening_fill_ratio": _num(g.opening_fill_ratio),
            }
            for g in result.gaps
        ],
        "doors": [
            {
                "id": d.id,
                "position": _point(d.position),
                "swing_end": _point(d.swing_end),
                "radius": _num(d.radius),
                "wall_id": d.wall_id,
                "swing_direction": d.swing_direction,
                "opens_into_room_id": d.opens_into_room_id,
                "swing_arc": [_point(point) for point in d.swing_arc],
                "confidence": _num(d.confidence),
            }
            for d in result.doors
        ],
        "windows": [
            {
                "id": w.id,
                "position": _point(w.position),
                "width": _num(w.width),
                "wall_id": w.wall_id,
            }
            for w in result.windows
        ],
        "rooms": [
            {
                "id": r.id,
                "polygon": [_point(p) for p in r.polygon],
                "area_px": _num(r.area),
                "label": r.label,
                "label_confidence": _num(r.label_confidence),
            }
            for r in result.rooms
        ],
        "debug": {
            "stage_timings": {k: _num(v) for k, v in result.debug.stage_timings.items()},
            "segment_counts": dict(result.debug.segment_counts),
            "roi_area_fraction": _num(result.debug.roi_area_fraction),
            "messages": list(result.debug.messages),
        },
    }


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def to_svg(
    result: CVTakeoffResult,
    image: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> str:
    width = result.metadata.image_width if result.metadata else 0
    height = result.metadata.image_height if result.metadata else 0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]

    if image is not None:
        ok, buf = cv2.imencode(".png", image)
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            parts.append(
                f'<image href="data:image/png;base64,{b64}" x="0" y="0" '
                f'width="{width}" height="{height}" opacity="0.4"/>'
            )

    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = " ".join(f"{p[0][0]},{p[0][1]}" for p in contour[::4])
            parts.append(
                f'<polygon points="{pts}" fill="none" stroke="#9467bd" '
                'stroke-width="2" stroke-dasharray="8 6"/>'
            )

    for room in result.rooms:
        pts = " ".join(f"{_num(p.x)},{_num(p.y)}" for p in room.polygon)
        parts.append(
            f'<polygon points="{pts}" fill="#aec7e8" fill-opacity="0.25" '
            'stroke="#7f7f7f"/>'
        )
        if room.label:
            cx = sum(p.x for p in room.polygon) / max(1, len(room.polygon))
            cy = sum(p.y for p in room.polygon) / max(1, len(room.polygon))
            parts.append(
                f'<text x="{_num(cx)}" y="{_num(cy)}" font-size="14" '
                f'fill="#333333" text-anchor="middle">{room.label}</text>'
            )

    for wall in result.walls:
        color = "#ff7f0e" if wall.merge_confidence < _LOW_CONFIDENCE else "#d62728"
        cl = wall.centerline
        parts.append(
            f'<line x1="{_num(cl.start.x)}" y1="{_num(cl.start.y)}" '
            f'x2="{_num(cl.end.x)}" y2="{_num(cl.end.y)}" stroke="{color}" '
            f'stroke-width="{_num(max(wall.thickness, wall.visual_thickness))}"/>'
        )

    for door in result.doors:
        p, s = door.position, door.swing_end
        if len(door.swing_arc) >= 2:
            points = " ".join(
                f"L {_num(point.x)} {_num(point.y)}" for point in door.swing_arc
            )
            parts.append(
                f'<path d="M {_num(p.x)} {_num(p.y)} {points} Z" '
                'fill="#2ca02c" fill-opacity="0.24" stroke="#2ca02c" '
                'stroke-width="1.5"/>'
            )
        parts.append(
            f'<line x1="{_num(p.x)}" y1="{_num(p.y)}" '
            f'x2="{_num(s.x)}" y2="{_num(s.y)}" '
            'stroke="#2ca02c" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle cx="{_num(p.x)}" cy="{_num(p.y)}" r="3" fill="#2ca02c"/>'
        )

    wall_lookup = {wall.id: wall for wall in result.walls}
    for wall in result.walls:
        for source_id in wall.source_ids:
            wall_lookup.setdefault(source_id, wall)
    for window in result.windows:
        p = window.position
        wall = wall_lookup.get(window.wall_id)
        if wall is None:
            continue
        cl = wall.centerline
        length = max(1e-6, cl.length)
        ux, uy = (cl.end.x - cl.start.x) / length, (cl.end.y - cl.start.y) / length
        half = window.width / 2.0
        thickness = max(wall.thickness, wall.visual_thickness)
        parts.append(
            f'<line x1="{_num(p.x - ux * half)}" y1="{_num(p.y - uy * half)}" '
            f'x2="{_num(p.x + ux * half)}" y2="{_num(p.y + uy * half)}" '
            f'stroke="#1f77b4" stroke-width="{_num(thickness)}"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)
