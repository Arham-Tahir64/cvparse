"""Versioned editable source of truth for a reviewed construction takeoff."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


MODEL_SCHEMA_VERSION = "2.0.0-alpha.1"


class ReviewStatus(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY_CORRECT = "likely_correct"
    NEEDS_REVIEW = "needs_review"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class ObjectSourceKind(str, Enum):
    AUTOMATIC_DETECTED = "automatic_detected"
    AUTOMATIC_INFERRED = "automatic_inferred"
    MANUAL_CREATED = "manual_created"
    MANUAL_ADJUSTED = "manual_adjusted"


class OpeningKind(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    ARCHWAY = "archway"
    UNKNOWN = "unknown"


class ScaleMethod(str, Enum):
    UNKNOWN = "unknown"
    MANUAL = "manual"
    VECTOR_DIMENSION = "vector_dimension"
    RASTER_DIMENSION = "raster_dimension"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class ApprovalStatus(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True)
class Coordinate:
    x: float
    y: float


@dataclass(slots=True)
class SourceEvidence:
    kind: ObjectSourceKind
    stage: str
    detector_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConfidenceBreakdown:
    overall: float
    geometry_quality: float | None = None
    association: float | None = None
    scale_plausibility: float | None = None
    symbol_evidence: float | None = None
    topology_consistency: float | None = None
    vector_raster_agreement: float | None = None
    duplicate_conflict: float | None = None


@dataclass(slots=True)
class ObjectMetadata:
    source: SourceEvidence
    confidence: ConfidenceBreakdown
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    locked: bool = False
    revision: int = 1


@dataclass(slots=True)
class PlanSource:
    id: str
    fingerprint: str
    source_path: str | None
    mime_type: str | None
    page_number: int
    image_width: int
    image_height: int
    dpi: int


@dataclass(slots=True)
class ScaleCalibration:
    pixels_per_unit: float | None = None
    unit: str = "ft"
    method: ScaleMethod = ScaleMethod.UNKNOWN
    confidence: float = 0.0
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW


@dataclass(slots=True)
class Node:
    id: str
    point: Coordinate
    connected_wall_ids: list[str]
    metadata: ObjectMetadata


@dataclass(slots=True)
class Wall:
    id: str
    start_node_id: str
    end_node_id: str
    start: Coordinate
    end: Coordinate
    polygon: list[Coordinate]
    thickness_px: float
    wall_type: str
    orientation: str
    connected_wall_ids: list[str]
    opening_ids: list[str]
    length_px: float
    length: float | None
    metadata: ObjectMetadata


@dataclass(slots=True)
class Opening:
    id: str
    wall_id: str
    kind: OpeningKind
    start_offset_px: float
    end_offset_px: float
    center: Coordinate
    width_px: float
    width: float | None
    orientation: str
    metadata: ObjectMetadata


@dataclass(slots=True)
class Door:
    id: str
    opening_id: str
    subtype: str
    swing_direction: str | None
    hinge_side: str | None
    hinge: Coordinate | None
    swing_end: Coordinate | None
    swing_arc: list[Coordinate]
    metadata: ObjectMetadata


@dataclass(slots=True)
class Window:
    id: str
    opening_id: str
    subtype: str
    sill_height: float | None
    metadata: ObjectMetadata


@dataclass(slots=True)
class Room:
    id: str
    polygon: list[Coordinate]
    label: str | None
    area_px: float
    area: float | None
    perimeter_px: float
    perimeter: float | None
    boundary_wall_ids: list[str]
    door_ids: list[str]
    window_ids: list[str]
    neighboring_room_ids: list[str]
    metadata: ObjectMetadata


@dataclass(slots=True)
class ValidationIssue:
    id: str
    code: str
    severity: IssueSeverity
    message: str
    affected_object_ids: list[str]
    uncertainty: float
    structural_impact: float
    cost_impact: float
    priority: float
    status: IssueStatus = IssueStatus.OPEN


@dataclass(slots=True)
class EditEvent:
    id: str
    action: str
    actor: str
    revision_before: int
    revision_after: int
    affected_object_ids: list[str]
    payload: dict[str, Any]
    timestamp: str


@dataclass(slots=True)
class TakeoffModel:
    id: str
    source: PlanSource
    scale: ScaleCalibration
    nodes: list[Node] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    edit_history: list[EditEvent] = field(default_factory=list)
    revision: int = 1
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    schema_version: str = MODEL_SCHEMA_VERSION
