"""Deterministic room-face reconstruction from the reviewed wall graph."""
from __future__ import annotations

import copy
import statistics
import uuid
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from .geometry import polygon_area, polygon_perimeter
from .models import (
    ConfidenceBreakdown,
    Coordinate,
    ObjectMetadata,
    ObjectSourceKind,
    ReviewStatus,
    Room,
    SourceEvidence,
    TakeoffModel,
)


class RoomTopologyError(ValueError):
    pass


@dataclass(slots=True)
class RoomFaceResult:
    rooms: list[Room]
    matched_room_ids: list[str]
    created_room_ids: list[str]
    removed_room_ids: list[str]


def _shape(points: list[Coordinate]) -> Polygon | None:
    if len(points) < 3:
        return None
    shape = Polygon([(point.x, point.y) for point in points])
    if not shape.is_valid:
        shape = shape.buffer(0)
    return shape if not shape.is_empty and shape.area > 0 else None


def _new_metadata() -> ObjectMetadata:
    return ObjectMetadata(
        source=SourceEvidence(
            kind=ObjectSourceKind.AUTOMATIC_INFERRED,
            stage="room_topology_recompute",
            details={"method": "shapely_polygonize"},
        ),
        confidence=ConfidenceBreakdown(
            overall=1.0, geometry_quality=1.0, topology_consistency=1.0,
        ),
        review_status=ReviewStatus.NEEDS_REVIEW,
    )


def _matched_metadata(room: Room) -> ObjectMetadata:
    metadata = copy.deepcopy(room.metadata)
    metadata.source.kind = ObjectSourceKind.MANUAL_ADJUSTED
    metadata.source.stage = "room_topology_recompute"
    metadata.source.details.pop("topology_invalidated_by_wall_ids", None)
    metadata.source.details["method"] = "shapely_polygonize"
    metadata.review_status = ReviewStatus.NEEDS_REVIEW
    metadata.locked = False
    metadata.revision += 1
    return metadata


def recompute_room_faces(model: TakeoffModel) -> RoomFaceResult:
    """Polygonize current non-rejected wall centerlines and preserve matched IDs."""
    walls = [
        wall for wall in model.walls
        if wall.metadata.review_status != ReviewStatus.REJECTED
    ]
    if len(walls) < 3:
        raise RoomTopologyError("at least three active walls are required")
    lines = {
        wall.id: LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])
        for wall in walls
    }
    network = unary_union(list(lines.values()))
    minimum_area = max(
        4.0,
        statistics.median(wall.thickness_px for wall in walls) ** 2,
    )
    faces = [face for face in polygonize(network) if face.area >= minimum_area]
    faces.sort(key=lambda face: (round(face.centroid.y, 6), round(face.centroid.x, 6)))
    if not faces:
        raise RoomTopologyError("reviewed wall graph contains no closed room faces")

    old_shapes = {
        room.id: shape for room in model.rooms
        if (shape := _shape(room.polygon)) is not None
    }
    candidates: list[tuple[float, str, int]] = []
    for index, face in enumerate(faces):
        for room_id, old in old_shapes.items():
            union_area = face.union(old).area
            iou = face.intersection(old).area / union_area if union_area else 0.0
            if iou >= 0.25:
                candidates.append((-iou, room_id, index))
    candidates.sort()
    match_by_face: dict[int, str] = {}
    used_old: set[str] = set()
    for _, room_id, index in candidates:
        if index in match_by_face or room_id in used_old:
            continue
        match_by_face[index] = room_id
        used_old.add(room_id)

    old_by_id = {room.id: room for room in model.rooms}
    openings_by_wall: dict[str, list[str]] = {}
    for opening in model.openings:
        openings_by_wall.setdefault(opening.wall_id, []).append(opening.id)
    door_by_opening = {door.opening_id: door.id for door in model.doors}
    window_by_opening = {window.opening_id: window.id for window in model.windows}
    rooms: list[Room] = []
    created: list[str] = []
    matched: list[str] = []
    for index, face in enumerate(faces):
        coordinates = [Coordinate(float(x), float(y)) for x, y in face.exterior.coords[:-1]]
        old_id = match_by_face.get(index)
        if old_id is None:
            room_id = f"room_manual_{uuid.uuid4().hex}"
            label = None
            metadata = _new_metadata()
            created.append(room_id)
        else:
            previous = old_by_id[old_id]
            room_id = old_id
            label = previous.label
            metadata = _matched_metadata(previous)
            matched.append(room_id)
        boundary = face.boundary
        boundary_wall_ids = sorted(
            wall_id for wall_id, line in lines.items()
            if boundary.intersection(line).length > 1e-6
        )
        opening_ids = {
            opening_id for wall_id in boundary_wall_ids
            for opening_id in openings_by_wall.get(wall_id, [])
        }
        area_px = polygon_area(coordinates)
        perimeter_px = polygon_perimeter(coordinates)
        scale = model.scale.pixels_per_unit
        rooms.append(Room(
            id=room_id,
            polygon=coordinates,
            label=label,
            area_px=area_px,
            area=area_px / (scale * scale) if scale else None,
            perimeter_px=perimeter_px,
            perimeter=perimeter_px / scale if scale else None,
            boundary_wall_ids=boundary_wall_ids,
            door_ids=sorted(
                door_by_opening[item] for item in opening_ids if item in door_by_opening
            ),
            window_ids=sorted(
                window_by_opening[item]
                for item in opening_ids if item in window_by_opening
            ),
            neighboring_room_ids=[],
            metadata=metadata,
        ))

    for room in rooms:
        room.neighboring_room_ids = sorted(
            other.id for other in rooms
            if other.id != room.id
            and set(room.boundary_wall_ids).intersection(other.boundary_wall_ids)
        )
    removed = sorted(set(old_by_id) - set(matched))
    return RoomFaceResult(
        rooms=rooms,
        matched_room_ids=sorted(matched),
        created_room_ids=sorted(created),
        removed_room_ids=removed,
    )
