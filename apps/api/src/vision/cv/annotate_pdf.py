"""Annotated-PDF renderer: draws all pipeline detections onto the plan.

For PDF inputs the annotations are drawn as vector overlays on the original
page; for raster inputs a new PDF page is created from the image first.
Every detection type gets its own color (matching the SVG overlay scheme):

- walls: red; walls with merge_confidence < 0.6: orange
- junctions: blue circles (radius by type)
- doors: green hinge dot + swing arc
- windows: dark orange tick across the wall
- door gaps: green dashed bbox; window gaps: orange dashed bbox
- rooms: light-blue fill with gray outline and label
- structural ROI boundary: purple dashed outline

Run from the repo root:
    python -m vision.cv.annotate_cli <input.(pdf|png|jpg)> <output.pdf>
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import fitz
import numpy as np

from .models import CVTakeoffResult

logger = logging.getLogger("flowbuildr.cv.annotate_pdf")

# color scheme (RGB 0-1 for PyMuPDF), one color per item type
COLORS = {
    "wall": (0.84, 0.15, 0.16),          # #d62728 red
    "wall_low_conf": (1.0, 0.50, 0.05),  # #ff7f0e orange
    "junction": (0.12, 0.47, 0.71),      # #1f77b4 blue
    "door": (0.17, 0.63, 0.17),          # #2ca02c green
    "window": (0.55, 0.34, 0.29),        # #8c564b brown-orange
    "gap_door": (0.09, 0.75, 0.81),      # #17becf cyan
    "gap_window": (0.89, 0.47, 0.76),    # #e377c2 pink
    "room_fill": (0.68, 0.78, 0.91),     # #aec7e8 light blue
    "room_stroke": (0.50, 0.50, 0.50),   # #7f7f7f gray
    "roi": (0.58, 0.40, 0.74),           # #9467bd purple
    "label": (0.10, 0.10, 0.10),
}

_LOW_CONFIDENCE = 0.6
_JUNCTION_RADIUS_PT = {
    "dead_end": 2.0, "L": 3.0, "T": 4.0, "X": 5.0, "Y": 5.0, "door_passage": 3.5,
}


def annotate_pdf_page(
    pdf_bytes: bytes,
    result: CVTakeoffResult,
    dpi: int,
    page_number: int = 0,
    roi_mask: Optional[np.ndarray] = None,
    junctions: Optional[list] = None,
) -> bytes:
    """Overlay detections on the original PDF page; returns single-page PDF bytes."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    doc = fitz.open()
    doc.insert_pdf(src, from_page=page_number, to_page=page_number)
    src.close()
    page = doc[0]
    _draw(page, result, scale=72.0 / dpi, roi_mask=roi_mask, junctions=junctions)
    out = doc.tobytes()
    doc.close()
    return out


def annotate_image_as_pdf(
    image: np.ndarray,
    result: CVTakeoffResult,
    dpi: int,
    roi_mask: Optional[np.ndarray] = None,
    junctions: Optional[list] = None,
) -> bytes:
    """Create a PDF page from a raster image and overlay detections."""
    h, w = image.shape[:2]
    scale = 72.0 / dpi
    doc = fitz.open()
    page = doc.new_page(width=w * scale, height=h * scale)
    ok, buf = cv2.imencode(".png", image)
    if ok:
        page.insert_image(page.rect, stream=buf.tobytes())
    _draw(page, result, scale=scale, roi_mask=roi_mask, junctions=junctions)
    out = doc.tobytes()
    doc.close()
    return out


def _draw(page, result: CVTakeoffResult, scale: float, roi_mask, junctions=None) -> None:
    shape = page.new_shape()

    def pt(x: float, y: float) -> fitz.Point:
        return fitz.Point(x * scale, y * scale)

    # rooms first so everything else draws on top of the fill
    for room in result.rooms:
        if len(room.polygon) < 3:
            continue
        points = [pt(p.x, p.y) for p in room.polygon]
        shape.draw_polyline(points + [points[0]])
        shape.finish(
            color=COLORS["room_stroke"], fill=COLORS["room_fill"],
            fill_opacity=0.25, width=1.0, closePath=True,
        )

    # structural ROI boundary
    if roi_mask is not None:
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            pts = [pt(float(p[0][0]), float(p[0][1])) for p in contour[::8]]
            if len(pts) >= 2:
                shape.draw_polyline(pts + [pts[0]])
                shape.finish(color=COLORS["roi"], width=1.2, dashes="[6 4] 0")

    # gap bounding boxes (dashed, under the wall strokes)
    for gap in result.gaps:
        rect = fitz.Rect(
            gap.bbox[0] * scale, gap.bbox[1] * scale,
            gap.bbox[2] * scale, gap.bbox[3] * scale,
        )
        color = COLORS["gap_door"] if gap.kind == "door" else COLORS["gap_window"]
        shape.draw_rect(rect)
        shape.finish(color=color, width=0.8, dashes="[3 3] 0")

    # walls
    for wall in result.walls:
        cl = wall.centerline
        low = wall.merge_confidence < _LOW_CONFIDENCE
        shape.draw_line(pt(cl.start.x, cl.start.y), pt(cl.end.x, cl.end.y))
        shape.finish(
            color=COLORS["wall_low_conf"] if low else COLORS["wall"],
            # PyMuPDF's width is the complete stroke width. visual_thickness
            # is likewise the complete measured wall band, not a radius.
            width=max(0.8, wall.visual_thickness * scale),
            stroke_opacity=0.85,
        )

    # junctions
    for junction in junctions or []:
        radius = _JUNCTION_RADIUS_PT.get(junction.junction_type, 3.0)
        shape.draw_circle(pt(junction.point.x, junction.point.y), radius)
        shape.finish(color=COLORS["junction"], width=1.0)

    # doors: hinge dot + chord to swing end + arc
    for door in result.doors:
        hinge = pt(door.position.x, door.position.y)
        swing = pt(door.swing_end.x, door.swing_end.y)
        shape.draw_circle(hinge, 2.5)
        shape.finish(color=COLORS["door"], fill=COLORS["door"], width=0.8)
        shape.draw_line(hinge, swing)
        shape.finish(color=COLORS["door"], width=1.2)
        shape.draw_circle(hinge, door.radius * scale)
        shape.finish(color=COLORS["door"], width=0.9, dashes="[2 2] 0")

    # windows: tick perpendicular-ish marker at the window position
    for window in result.windows:
        half = window.width * scale / 2.0
        c = pt(window.position.x, window.position.y)
        shape.draw_line(fitz.Point(c.x - half, c.y), fitz.Point(c.x + half, c.y))
        shape.finish(color=COLORS["window"], width=2.5, stroke_opacity=0.9)
        shape.draw_circle(c, 2.0)
        shape.finish(color=COLORS["window"], fill=COLORS["window"], width=0.5)

    shape.commit()

    # room labels and legend as text (separate insert calls)
    for room in result.rooms:
        if not room.label or not room.polygon:
            continue
        cx = sum(p.x for p in room.polygon) / len(room.polygon) * scale
        cy = sum(p.y for p in room.polygon) / len(room.polygon) * scale
        page.insert_text(
            fitz.Point(cx, cy), room.label, fontsize=9,
            color=COLORS["label"],
        )

    _draw_legend(page, result)


def _draw_legend(page, result: CVTakeoffResult) -> None:
    entries = [
        ("wall", f"walls ({len(result.walls)})", COLORS["wall"]),
        ("wall_low", "walls, low confidence", COLORS["wall_low_conf"]),
        ("door", f"doors ({len(result.doors)})", COLORS["door"]),
        ("window", f"windows ({len(result.windows)})", COLORS["window"]),
        ("gap_door", "door gaps", COLORS["gap_door"]),
        ("gap_window", "window gaps", COLORS["gap_window"]),
        ("room", f"rooms ({len(result.rooms)})", COLORS["room_fill"]),
        ("junction", "junctions", COLORS["junction"]),
        ("roi", "structural ROI", COLORS["roi"]),
    ]
    x, y = 10.0, 12.0
    line_h = 11.0
    box = fitz.Rect(x - 4, y - 9, x + 150, y + line_h * len(entries))
    shape = page.new_shape()
    shape.draw_rect(box)
    shape.finish(color=(0.3, 0.3, 0.3), fill=(1, 1, 1), fill_opacity=0.85, width=0.5)
    for i, (_, _, color) in enumerate(entries):
        cy = y + i * line_h - 2.5
        shape.draw_line(fitz.Point(x, cy), fitz.Point(x + 14, cy))
        shape.finish(color=color, width=3.0)
    shape.commit()
    for i, (_, label, _) in enumerate(entries):
        page.insert_text(fitz.Point(x + 18, y + i * line_h), label, fontsize=7,
                         color=(0.1, 0.1, 0.1))
