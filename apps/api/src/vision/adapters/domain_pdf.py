"""Combined reviewed-PDF rendering from source asset plus authoritative model."""
from __future__ import annotations

import cv2
import fitz
import numpy as np

from vision.domain.geometry import point_at_offset
from vision.domain.models import OpeningKind, ReviewStatus, TakeoffModel


class DomainRenderError(ValueError):
    pass


_COLORS = {
    "wall": (0.84, 0.15, 0.16),
    "wall_low": (1.0, 0.50, 0.05),
    "door": (0.17, 0.63, 0.17),
    "window": (0.12, 0.47, 0.71),
    "room_fill": (0.68, 0.78, 0.91),
    "room_stroke": (0.50, 0.50, 0.50),
}


def _visible(item) -> bool:
    return item.metadata.review_status != ReviewStatus.REJECTED


def _new_document(
    content: bytes, mime_type: str | None, model: TakeoffModel,
) -> tuple[fitz.Document, fitz.Page, float]:
    dpi = model.source.dpi or 200
    scale = 72.0 / dpi
    if mime_type == "application/pdf":
        try:
            source = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DomainRenderError("persisted source is not a readable PDF") from exc
        page_number = model.source.page_number
        if page_number < 0 or page_number >= source.page_count:
            source.close()
            raise DomainRenderError(f"source PDF has no page {page_number}")
        document = fitz.open()
        document.insert_pdf(source, from_page=page_number, to_page=page_number)
        source.close()
        return document, document[0], scale

    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DomainRenderError("persisted source is not a readable raster image")
    height, width = image.shape[:2]
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise DomainRenderError("persisted raster image could not be encoded")
    document = fitz.open()
    page = document.new_page(width=width * scale, height=height * scale)
    page.insert_image(page.rect, stream=encoded.tobytes())
    return document, page, scale


def _draw_model(page: fitz.Page, model: TakeoffModel, scale: float) -> None:
    shape = page.new_shape()

    def point(value) -> fitz.Point:
        return fitz.Point(value.x * scale, value.y * scale)

    for room in model.rooms:
        if not _visible(room) or len(room.polygon) < 3:
            continue
        points = [point(item) for item in room.polygon]
        shape.draw_polyline(points + [points[0]])
        shape.finish(
            color=_COLORS["room_stroke"], fill=_COLORS["room_fill"],
            fill_opacity=0.25, width=0.8, closePath=True,
        )

    walls = {wall.id: wall for wall in model.walls if _visible(wall)}
    for wall in walls.values():
        if len(wall.polygon) < 3:
            continue
        points = [point(item) for item in wall.polygon]
        color = (
            _COLORS["wall_low"]
            if wall.metadata.confidence.overall < 0.6 else _COLORS["wall"]
        )
        shape.draw_polyline(points + [points[0]])
        shape.finish(
            color=color, fill=color, width=0.2,
            stroke_opacity=0.85, fill_opacity=0.85, closePath=True,
        )

    openings = {
        opening.id: opening for opening in model.openings
        if _visible(opening) and opening.wall_id in walls
    }
    for opening in openings.values():
        wall = walls[opening.wall_id]
        start = point_at_offset(wall.start, wall.end, opening.start_offset_px)
        end = point_at_offset(wall.start, wall.end, opening.end_offset_px)
        color = (
            _COLORS["door"] if opening.kind == OpeningKind.DOOR
            else _COLORS["window"]
        )
        shape.draw_line(point(start), point(end))
        shape.finish(
            color=color,
            width=max(1.5, wall.thickness_px * scale),
            stroke_opacity=0.72,
        )

    for door in model.doors:
        if not _visible(door) or door.opening_id not in openings:
            continue
        if door.hinge is not None and door.swing_end is not None:
            shape.draw_line(point(door.hinge), point(door.swing_end))
            shape.finish(color=_COLORS["door"], width=1.2)
        if len(door.swing_arc) >= 2:
            shape.draw_polyline([point(item) for item in door.swing_arc])
            shape.finish(color=_COLORS["door"], width=0.9)

    shape.commit()


def render_reviewed_pdf(
    source_content: bytes,
    mime_type: str | None,
    model: TakeoffModel,
) -> bytes:
    """Return one source page with overlays from exactly `model.revision`."""
    document, page, scale = _new_document(source_content, mime_type, model)
    _draw_model(page, model, scale)
    metadata = document.metadata
    metadata.update({
        "title": "FlowBuildr reviewed takeoff",
        "subject": f"Model {model.id} revision {model.revision}",
        "keywords": f"flowbuildr,reviewed,model-revision-{model.revision}",
    })
    document.set_metadata(metadata)
    output = document.tobytes(garbage=3, deflate=True)
    document.close()
    return output
