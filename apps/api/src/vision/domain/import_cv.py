"""Compatibility importer from ephemeral CV results to the editable model."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from vision.cv.models import CVTakeoffResult, Point as CVPoint

from .models import (
    ConfidenceBreakdown,
    Coordinate,
    Door,
    Node,
    ObjectMetadata,
    ObjectSourceKind,
    Opening,
    OpeningKind,
    PlanSource,
    ReviewStatus,
    Room,
    ScaleCalibration,
    SourceEvidence,
    TakeoffModel,
    Wall,
    Window,
)
from .validation import validate_model


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _coordinate(point: CVPoint) -> Coordinate:
    return Coordinate(round(float(point.x), 3), round(float(point.y), 3))


def _point_key(point: Coordinate) -> tuple[float, float]:
    return round(point.x, 3), round(point.y, 3)


def _clamp_confidence(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _metadata(
    kind: ObjectSourceKind,
    stage: str,
    detector_ids: list[str],
    overall: float,
    *,
    geometry_quality: float | None = None,
    association: float | None = None,
    symbol_evidence: float | None = None,
    topology_consistency: float | None = None,
    details: dict | None = None,
) -> ObjectMetadata:
    overall = _clamp_confidence(overall)
    status = (
        ReviewStatus.LIKELY_CORRECT
        if overall >= 0.75
        else ReviewStatus.NEEDS_REVIEW
    )
    return ObjectMetadata(
        source=SourceEvidence(
            kind=kind, stage=stage,
            detector_ids=sorted(set(identifier for identifier in detector_ids if identifier)),
            details=dict(details or {}),
        ),
        confidence=ConfidenceBreakdown(
            overall=overall,
            geometry_quality=(
                _clamp_confidence(geometry_quality)
                if geometry_quality is not None else None
            ),
            association=(
                _clamp_confidence(association) if association is not None else None
            ),
            symbol_evidence=(
                _clamp_confidence(symbol_evidence)
                if symbol_evidence is not None else None
            ),
            topology_consistency=(
                _clamp_confidence(topology_consistency)
                if topology_consistency is not None else None
            ),
        ),
        review_status=status,
    )


def _source_fingerprint(
    result: CVTakeoffResult, supplied: str | None,
) -> str:
    if supplied:
        return supplied.lower()
    meta = result.metadata
    fallback = "|".join([
        str(meta.source_path if meta else ""),
        str(meta.image_width if meta else 0),
        str(meta.image_height if meta else 0),
        str(meta.dpi if meta else 0),
        str(meta.page_number if meta else 0),
    ])
    return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def _wall_polygon(start: Coordinate, end: Coordinate, thickness: float) -> list[Coordinate]:
    dx, dy = end.x - start.x, end.y - start.y
    length = max(1e-9, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    half = max(0.0, thickness) / 2.0
    return [
        Coordinate(start.x + nx * half, start.y + ny * half),
        Coordinate(end.x + nx * half, end.y + ny * half),
        Coordinate(end.x - nx * half, end.y - ny * half),
        Coordinate(start.x - nx * half, start.y - ny * half),
    ]


def _distance_to_wall(point: Coordinate, wall: Wall) -> float:
    dx, dy = wall.end.x - wall.start.x, wall.end.y - wall.start.y
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return math.hypot(point.x - wall.start.x, point.y - wall.start.y)
    t = ((point.x - wall.start.x) * dx + (point.y - wall.start.y) * dy) / denominator
    t = min(1.0, max(0.0, t))
    x, y = wall.start.x + t * dx, wall.start.y + t * dy
    return math.hypot(point.x - x, point.y - y)


def _offset_on_wall(point: Coordinate, wall: Wall) -> float:
    dx, dy = wall.end.x - wall.start.x, wall.end.y - wall.start.y
    length = max(1e-9, wall.length_px)
    return ((point.x - wall.start.x) * dx + (point.y - wall.start.y) * dy) / length


def _room_perimeter(polygon: list[Coordinate]) -> float:
    if len(polygon) < 2:
        return 0.0
    points = polygon + [polygon[0]]
    return sum(
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(points, points[1:])
    )


def import_cv_result(
    result: CVTakeoffResult,
    *,
    source_fingerprint: str | None = None,
    mime_type: str | None = None,
) -> TakeoffModel:
    """Import automatic candidates without treating them as confirmed data.

    IDs are deterministic for an identical source fingerprint and geometry.
    The fingerprint should be the upload's SHA-256 in production; the metadata
    fallback exists for direct library callers and tests.
    """
    meta = result.metadata
    fingerprint = _source_fingerprint(result, source_fingerprint)
    page_number = meta.page_number if meta else 0
    model_id = _stable_id("plan", fingerprint, page_number)
    source = PlanSource(
        id=_stable_id("source", fingerprint), fingerprint=fingerprint,
        source_path=meta.source_path if meta else None, mime_type=mime_type,
        page_number=page_number,
        image_width=meta.image_width if meta else 0,
        image_height=meta.image_height if meta else 0,
        dpi=meta.dpi if meta else 0,
    )

    node_by_key: dict[tuple[float, float], Node] = {}
    detector_wall_ids: dict[str, list[str]] = defaultdict(list)
    walls: list[Wall] = []
    wall_signature_counts: dict[tuple, int] = defaultdict(int)

    def node_for(point: Coordinate) -> Node:
        key = _point_key(point)
        if key not in node_by_key:
            node_by_key[key] = Node(
                id=_stable_id("node", model_id, key[0], key[1]),
                point=point, connected_wall_ids=[],
                metadata=_metadata(
                    ObjectSourceKind.AUTOMATIC_INFERRED,
                    "09_junction_snapping", [], 0.7,
                    geometry_quality=0.7, topology_consistency=0.7,
                ),
            )
        return node_by_key[key]

    for candidate in result.walls:
        start = _coordinate(candidate.centerline.start)
        end = _coordinate(candidate.centerline.end)
        start_node, end_node = node_for(start), node_for(end)
        thickness = max(float(candidate.thickness), float(candidate.visual_thickness), 0.0)
        signature = (
            min(_point_key(start), _point_key(end)),
            max(_point_key(start), _point_key(end)),
            round(thickness, 3), candidate.orientation,
        )
        occurrence = wall_signature_counts[signature]
        wall_signature_counts[signature] += 1
        wall_id = _stable_id("wall", model_id, signature, occurrence)
        confidence = _clamp_confidence(candidate.merge_confidence)
        wall = Wall(
            id=wall_id, start_node_id=start_node.id, end_node_id=end_node.id,
            start=start, end=end,
            polygon=_wall_polygon(start, end, thickness),
            thickness_px=thickness, wall_type=candidate.wall_type,
            orientation=candidate.orientation, connected_wall_ids=[], opening_ids=[],
            length_px=float(candidate.centerline.length), length=candidate.length_ft,
            metadata=_metadata(
                ObjectSourceKind.AUTOMATIC_DETECTED, "08_wall_extraction",
                [candidate.id, *candidate.source_ids], confidence,
                geometry_quality=candidate.fit_support_ratio,
                association=candidate.merge_confidence,
                details={
                    "merge_kind": candidate.merge_kind,
                    "fit_support_ratio": candidate.fit_support_ratio,
                    "merge_confidence": candidate.merge_confidence,
                    "detector_length_px": candidate.length_px,
                },
            ),
        )
        walls.append(wall)
        start_node.connected_wall_ids.append(wall_id)
        end_node.connected_wall_ids.append(wall_id)
        for detector_id in [candidate.id, *candidate.source_ids]:
            if detector_id:
                detector_wall_ids[detector_id].append(wall_id)

    wall_by_id = {wall.id: wall for wall in walls}
    for node in node_by_key.values():
        node.connected_wall_ids.sort()
        for wall_id in node.connected_wall_ids:
            wall_by_id[wall_id].connected_wall_ids.extend(
                other for other in node.connected_wall_ids if other != wall_id
            )
    for wall in walls:
        wall.connected_wall_ids = sorted(set(wall.connected_wall_ids))

    def resolve_wall(detector_id: str, point: Coordinate) -> Wall | None:
        candidate_ids = detector_wall_ids.get(detector_id, [])
        candidates = [wall_by_id[item] for item in candidate_ids]
        if not candidates:
            candidates = walls
        return min(candidates, key=lambda item: _distance_to_wall(point, item), default=None)

    openings: list[Opening] = []
    opening_signature_counts: dict[tuple, int] = defaultdict(int)

    def add_opening(
        detector_id: str,
        detector_wall_id: str,
        point: Coordinate,
        width: float,
        kind: OpeningKind,
        stage: str,
        confidence: float,
        details: dict | None = None,
    ) -> Opening:
        wall = resolve_wall(detector_wall_id, point)
        wall_id = wall.id if wall is not None else detector_wall_id
        center_offset = _offset_on_wall(point, wall) if wall is not None else 0.0
        start_offset = center_offset - width / 2.0
        end_offset = center_offset + width / 2.0
        signature = (
            wall_id, kind.value, round(start_offset, 3), round(end_offset, 3),
        )
        occurrence = opening_signature_counts[signature]
        opening_signature_counts[signature] += 1
        opening = Opening(
            id=_stable_id("opening", model_id, signature, occurrence), wall_id=wall_id,
            kind=kind, start_offset_px=start_offset, end_offset_px=end_offset,
            center=point, width_px=float(width), width=None,
            orientation=wall.orientation if wall is not None else "unknown",
            metadata=_metadata(
                ObjectSourceKind.AUTOMATIC_DETECTED, stage, [detector_id], confidence,
                geometry_quality=confidence,
                association=(1.0 if wall is not None else 0.0), details=details,
            ),
        )
        openings.append(opening)
        if wall is not None:
            wall.opening_ids.append(opening.id)
        return opening

    for gap in result.gaps:
        kind = OpeningKind.DOOR if gap.kind == "door" else OpeningKind.WINDOW
        add_opening(
            gap.id, gap.wall_id, _coordinate(gap.center), float(gap.width_px),
            kind, "10_door_detection" if kind == OpeningKind.DOOR else "12_window_detection",
            _clamp_confidence(gap.wall_break_score),
            details={
                "bbox": list(gap.bbox),
                "wall_break_score": gap.wall_break_score,
                "opening_fill_ratio": gap.opening_fill_ratio,
            },
        )

    def match_opening(kind: OpeningKind, detector_wall_id: str, point: Coordinate, width: float):
        wall = resolve_wall(detector_wall_id, point)
        wall_id = wall.id if wall is not None else detector_wall_id
        candidates = [
            opening for opening in openings
            if opening.kind == kind and opening.wall_id == wall_id
        ]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda opening: math.hypot(
                opening.center.x - point.x, opening.center.y - point.y,
            ),
        )
        distance = math.hypot(best.center.x - point.x, best.center.y - point.y)
        return best if distance <= max(width, best.width_px) else None

    doors: list[Door] = []
    for candidate in result.doors:
        hinge = _coordinate(candidate.position)
        opening = match_opening(
            OpeningKind.DOOR, candidate.wall_id, hinge, float(candidate.radius),
        )
        if opening is None:
            opening = add_opening(
                candidate.id, candidate.wall_id, hinge, float(candidate.radius),
                OpeningKind.DOOR, "10_door_detection", candidate.confidence,
                details={"synthesized_from": "door_without_gap"},
            )
        confidence = _clamp_confidence(candidate.confidence)
        doors.append(Door(
            id=_stable_id("door", model_id, opening.id), opening_id=opening.id,
            subtype="unknown", swing_direction=candidate.swing_direction,
            hinge_side=None, hinge=hinge, swing_end=_coordinate(candidate.swing_end),
            swing_arc=[_coordinate(point) for point in candidate.swing_arc],
            metadata=_metadata(
                ObjectSourceKind.AUTOMATIC_DETECTED, "10_door_detection",
                [candidate.id], confidence, symbol_evidence=confidence,
                association=1.0 if opening.wall_id in wall_by_id else 0.0,
            ),
        ))

    windows: list[Window] = []
    for candidate in result.windows:
        center = _coordinate(candidate.position)
        opening = match_opening(
            OpeningKind.WINDOW, candidate.wall_id, center, float(candidate.width),
        )
        if opening is None:
            opening = add_opening(
                candidate.id, candidate.wall_id, center, float(candidate.width),
                OpeningKind.WINDOW, "12_window_detection", 0.5,
                details={"synthesized_from": "window_without_gap"},
            )
        windows.append(Window(
            id=_stable_id("window", model_id, opening.id), opening_id=opening.id,
            subtype="unknown", sill_height=None,
            metadata=_metadata(
                ObjectSourceKind.AUTOMATIC_DETECTED, "12_window_detection",
                [candidate.id], 0.5, association=1.0,
            ),
        ))

    rooms: list[Room] = []
    room_signature_counts: dict[tuple, int] = defaultdict(int)
    for candidate in result.rooms:
        polygon = [_coordinate(point) for point in candidate.polygon]
        signature = (
            tuple(_point_key(point) for point in polygon), candidate.label,
            round(float(candidate.area), 3),
        )
        occurrence = room_signature_counts[signature]
        room_signature_counts[signature] += 1
        rooms.append(Room(
            id=_stable_id("room", model_id, signature, occurrence),
            polygon=polygon, label=candidate.label, area_px=float(candidate.area),
            area=None, perimeter_px=_room_perimeter(polygon), perimeter=None,
            boundary_wall_ids=[], door_ids=[], window_ids=[],
            neighboring_room_ids=[],
            metadata=_metadata(
                ObjectSourceKind.AUTOMATIC_INFERRED, "11_room_extraction",
                [candidate.id], 0.5, geometry_quality=0.5,
                symbol_evidence=candidate.label_confidence,
                details={"label_confidence": candidate.label_confidence},
            ),
        ))

    for wall in walls:
        wall.opening_ids.sort()

    model = TakeoffModel(
        id=model_id, source=source, scale=ScaleCalibration(),
        nodes=sorted(node_by_key.values(), key=lambda item: item.id),
        walls=walls, openings=openings, doors=doors, windows=windows, rooms=rooms,
    )
    model.validation_issues = validate_model(model)
    return model
