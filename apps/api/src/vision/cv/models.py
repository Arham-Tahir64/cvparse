"""All data types for the CV floor plan pipeline.

Value types are frozen dataclasses; entity types are mutable with slots.
Coordinate system: origin top-left, x right, y down, pixels at working DPI.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base error for pipeline failures. Carries the module name."""

    def __init__(self, module: str, message: str):
        self.module = module
        self.message = message
        super().__init__(f"[{module}] {message}")


class NoLinesDetectedError(PipelineError):
    pass


class NoWallsFoundError(PipelineError):
    pass


class NoRoomsExtractedError(PipelineError):
    pass


class StructuralROIError(PipelineError):
    pass


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


LineClassification = Literal[
    "unknown", "wall", "dimension", "grid", "hatch", "leader", "text_baseline"
]


@dataclass(frozen=True, slots=True)
class LineSegment:
    start: Point
    end: Point
    thickness: float = 1.0
    classification: LineClassification = "unknown"
    id: str = ""

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def angle_rad(self) -> float:
        """Direction-independent angle in [0, pi)."""
        angle = math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
        return angle % math.pi

    @property
    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2.0, (self.start.y + self.end.y) / 2.0)

    @property
    def is_horizontal(self) -> bool:
        a = self.angle_rad
        return a < math.radians(10) or a > math.pi - math.radians(10)

    @property
    def is_vertical(self) -> bool:
        return abs(self.angle_rad - math.pi / 2) < math.radians(10)


@dataclass(frozen=True, slots=True)
class TextElement:
    text: str
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    confidence: float

    @property
    def center(self) -> Point:
        return Point((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Wall:
    id: str
    orientation: Literal["H", "V", "diagonal"]
    centerline: LineSegment
    thickness: float
    visual_thickness: float = 0.0
    wall_type: Literal["exterior", "interior", "unknown"] = "unknown"
    merge_kind: Literal["paired_faces", "single_face"] = "paired_faces"
    fit_support_ratio: float = 1.0
    merge_confidence: float = 1.0
    source_ids: list[str] = field(default_factory=list)
    length_px: float = 0.0
    length_ft: Optional[float] = None


@dataclass(slots=True)
class Junction:
    id: str
    point: Point
    walls: list[str] = field(default_factory=list)
    junction_type: Literal["dead_end", "L", "T", "X", "Y", "door_passage"] = "dead_end"


@dataclass(slots=True)
class Gap:
    id: str
    wall_id: str
    orientation: Literal["H", "V"]
    center: Point
    width_px: float
    bbox: tuple[float, float, float, float]
    kind: Literal["door", "window"]
    wall_break_score: float
    opening_fill_ratio: float


@dataclass(slots=True)
class Door:
    id: str
    position: Point  # hinge
    swing_end: Point
    radius: float
    wall_id: str
    swing_direction: Literal["cw", "ccw"]
    opens_into_room_id: Optional[str] = None


@dataclass(slots=True)
class Window:
    id: str
    position: Point
    width: float
    wall_id: str


@dataclass(slots=True)
class Room:
    id: str
    polygon: list[Point]
    label: Optional[str] = None
    label_confidence: float = 0.0
    area: float = 0.0


# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PlanMetadata:
    source_path: Optional[str]
    image_width: int
    image_height: int
    dpi: int
    page_number: int
    wall_count: int
    room_count: int


@dataclass(slots=True)
class DebugInfo:
    stage_timings: dict[str, float] = field(default_factory=dict)
    segment_counts: dict[str, int] = field(default_factory=dict)
    roi_area_fraction: float = 0.0
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CVTakeoffResult:
    walls: list[Wall] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    metadata: PlanMetadata = None  # type: ignore[assignment]
    debug: DebugInfo = field(default_factory=DebugInfo)
    preview_image: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineState:
    config: "PipelineConfig"
    mime_type: str = "image/png"
    source_path: Optional[str] = None
    image: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    binary_masked: Optional[np.ndarray] = None
    structural_roi_mask: Optional[np.ndarray] = None
    # hole-filled plan component BEFORE dilation: tight wall-mass coverage,
    # used by module 04 to flag thin lines outside the plan body
    structural_core_mask: Optional[np.ndarray] = None
    dpi: int = 200
    page_number: int = 0
    raw_lines: list[LineSegment] = field(default_factory=list)
    raw_texts: list[TextElement] = field(default_factory=list)
    classified_lines: list[LineSegment] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    debug: DebugInfo = field(default_factory=DebugInfo)

    def to_takeoff_result(self) -> CVTakeoffResult:
        height, width = (self.image.shape[:2] if self.image is not None else (0, 0))
        metadata = PlanMetadata(
            source_path=self.source_path,
            image_width=int(width),
            image_height=int(height),
            dpi=self.dpi,
            page_number=self.page_number,
            wall_count=len(self.walls),
            room_count=len(self.rooms),
        )
        return CVTakeoffResult(
            walls=list(self.walls),
            gaps=list(self.gaps),
            doors=list(self.doors),
            windows=list(self.windows),
            rooms=list(self.rooms),
            metadata=metadata,
            debug=self.debug,
        )


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

class IdGenerator:
    """Yields IDs like W0001, W0002, ... for a given prefix."""

    def __init__(self, prefix: str, start: int = 1):
        self.prefix = prefix
        self._counter = itertools.count(start)

    def __call__(self) -> str:
        return f"{self.prefix}{next(self._counter):04d}"

    def next(self) -> str:
        return self()
