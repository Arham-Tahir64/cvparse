"""Lossless JSON serialization for the editable takeoff model."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from .models import (
    ApprovalStatus,
    ConfidenceBreakdown,
    Coordinate,
    Door,
    IssueSeverity,
    IssueStatus,
    Node,
    ObjectMetadata,
    ObjectSourceKind,
    Opening,
    OpeningKind,
    PlanSource,
    ReviewStatus,
    Room,
    ScaleCalibration,
    ScaleMethod,
    SourceEvidence,
    TakeoffModel,
    ValidationIssue,
    Wall,
    Window,
)


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _encode(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def to_json_dict(model: TakeoffModel) -> dict[str, Any]:
    """Return a JSON-safe representation with an explicit model schema."""
    return _encode(model)


def _coordinate(data: dict | None) -> Coordinate | None:
    if data is None:
        return None
    return Coordinate(x=float(data["x"]), y=float(data["y"]))


def _source(data: dict) -> SourceEvidence:
    return SourceEvidence(
        kind=ObjectSourceKind(data["kind"]), stage=str(data["stage"]),
        detector_ids=list(data.get("detector_ids", [])),
        details=dict(data.get("details", {})),
    )


def _confidence(data: dict) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        overall=float(data["overall"]),
        geometry_quality=data.get("geometry_quality"),
        association=data.get("association"),
        scale_plausibility=data.get("scale_plausibility"),
        symbol_evidence=data.get("symbol_evidence"),
        topology_consistency=data.get("topology_consistency"),
        vector_raster_agreement=data.get("vector_raster_agreement"),
        duplicate_conflict=data.get("duplicate_conflict"),
    )


def _metadata(data: dict) -> ObjectMetadata:
    return ObjectMetadata(
        source=_source(data["source"]),
        confidence=_confidence(data["confidence"]),
        review_status=ReviewStatus(data["review_status"]),
        locked=bool(data.get("locked", False)),
        revision=int(data.get("revision", 1)),
    )


def from_json_dict(data: dict[str, Any]) -> TakeoffModel:
    """Rehydrate a model exactly enough for editing and validation."""
    source_data = data["source"]
    source = PlanSource(
        id=source_data["id"], fingerprint=source_data["fingerprint"],
        source_path=source_data.get("source_path"),
        mime_type=source_data.get("mime_type"),
        page_number=int(source_data["page_number"]),
        image_width=int(source_data["image_width"]),
        image_height=int(source_data["image_height"]),
        dpi=int(source_data["dpi"]),
    )
    scale_data = data["scale"]
    scale = ScaleCalibration(
        pixels_per_unit=scale_data.get("pixels_per_unit"),
        unit=scale_data.get("unit", "ft"),
        method=ScaleMethod(scale_data.get("method", ScaleMethod.UNKNOWN.value)),
        confidence=float(scale_data.get("confidence", 0.0)),
        review_status=ReviewStatus(
            scale_data.get("review_status", ReviewStatus.NEEDS_REVIEW.value)
        ),
    )

    nodes = [Node(
        id=item["id"], point=_coordinate(item["point"]),
        connected_wall_ids=list(item.get("connected_wall_ids", [])),
        metadata=_metadata(item["metadata"]),
    ) for item in data.get("nodes", [])]
    walls = [Wall(
        id=item["id"], start_node_id=item["start_node_id"],
        end_node_id=item["end_node_id"], start=_coordinate(item["start"]),
        end=_coordinate(item["end"]),
        polygon=[_coordinate(point) for point in item.get("polygon", [])],
        thickness_px=float(item["thickness_px"]), wall_type=item["wall_type"],
        orientation=item["orientation"],
        connected_wall_ids=list(item.get("connected_wall_ids", [])),
        opening_ids=list(item.get("opening_ids", [])),
        length_px=float(item["length_px"]), length=item.get("length"),
        metadata=_metadata(item["metadata"]),
    ) for item in data.get("walls", [])]
    openings = [Opening(
        id=item["id"], wall_id=item["wall_id"], kind=OpeningKind(item["kind"]),
        start_offset_px=float(item["start_offset_px"]),
        end_offset_px=float(item["end_offset_px"]),
        center=_coordinate(item["center"]), width_px=float(item["width_px"]),
        width=item.get("width"), orientation=item["orientation"],
        metadata=_metadata(item["metadata"]),
    ) for item in data.get("openings", [])]
    doors = [Door(
        id=item["id"], opening_id=item["opening_id"], subtype=item["subtype"],
        swing_direction=item.get("swing_direction"), hinge_side=item.get("hinge_side"),
        hinge=_coordinate(item.get("hinge")),
        swing_end=_coordinate(item.get("swing_end")),
        swing_arc=[_coordinate(point) for point in item.get("swing_arc", [])],
        metadata=_metadata(item["metadata"]),
    ) for item in data.get("doors", [])]
    windows = [Window(
        id=item["id"], opening_id=item["opening_id"], subtype=item["subtype"],
        sill_height=item.get("sill_height"), metadata=_metadata(item["metadata"]),
    ) for item in data.get("windows", [])]
    rooms = [Room(
        id=item["id"],
        polygon=[_coordinate(point) for point in item.get("polygon", [])],
        label=item.get("label"), area_px=float(item["area_px"]),
        area=item.get("area"), perimeter_px=float(item["perimeter_px"]),
        perimeter=item.get("perimeter"),
        boundary_wall_ids=list(item.get("boundary_wall_ids", [])),
        door_ids=list(item.get("door_ids", [])),
        window_ids=list(item.get("window_ids", [])),
        neighboring_room_ids=list(item.get("neighboring_room_ids", [])),
        metadata=_metadata(item["metadata"]),
    ) for item in data.get("rooms", [])]
    issues = [ValidationIssue(
        id=item["id"], code=item["code"], severity=IssueSeverity(item["severity"]),
        message=item["message"],
        affected_object_ids=list(item.get("affected_object_ids", [])),
        uncertainty=float(item["uncertainty"]),
        structural_impact=float(item["structural_impact"]),
        cost_impact=float(item["cost_impact"]), priority=float(item["priority"]),
        status=IssueStatus(item.get("status", IssueStatus.OPEN.value)),
    ) for item in data.get("validation_issues", [])]
    return TakeoffModel(
        id=data["id"], source=source, scale=scale, nodes=nodes, walls=walls,
        openings=openings, doors=doors, windows=windows, rooms=rooms,
        validation_issues=issues, revision=int(data.get("revision", 1)),
        approval_status=ApprovalStatus(
            data.get("approval_status", ApprovalStatus.DRAFT.value)
        ),
        schema_version=data["schema_version"],
    )

