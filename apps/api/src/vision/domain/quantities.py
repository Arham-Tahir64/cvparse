"""Model-native takeoff quantities with explicit review inclusion semantics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .models import IssueSeverity, ReviewStatus, TakeoffModel


class QuantityBasis(str, Enum):
    PROVISIONAL = "provisional"
    VERIFIED = "verified"


@dataclass(slots=True)
class QuantitySummary:
    model_id: str
    model_revision: int
    basis: QuantityBasis
    unit: str
    scale_confirmed: bool
    complete: bool
    authoritative: bool
    counts: dict[str, int]
    pixel_measurements: dict[str, float]
    calibrated_measurements: dict[str, float | None]
    included_object_ids: list[str] = field(default_factory=list)
    excluded_object_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["basis"] = self.basis.value
        return data


def _accepted(status: ReviewStatus, basis: QuantityBasis) -> bool:
    if status == ReviewStatus.REJECTED:
        return False
    return basis == QuantityBasis.PROVISIONAL or status == ReviewStatus.CONFIRMED


def calculate_quantities(
    model: TakeoffModel,
    basis: QuantityBasis = QuantityBasis.PROVISIONAL,
) -> QuantitySummary:
    """Calculate one revision without promoting uncertain candidates to final totals."""
    scale = model.scale.pixels_per_unit
    scale_confirmed = (
        scale is not None
        and scale > 0
        and model.scale.review_status == ReviewStatus.CONFIRMED
    )
    walls = {
        wall.id: wall for wall in model.walls
        if _accepted(wall.metadata.review_status, basis)
    }
    openings = {
        opening.id: opening for opening in model.openings
        if _accepted(opening.metadata.review_status, basis)
        and opening.wall_id in walls
    }
    rooms = [
        room for room in model.rooms
        if _accepted(room.metadata.review_status, basis)
    ]
    doors = [
        door for door in model.doors
        if _accepted(door.metadata.review_status, basis)
        and door.opening_id in openings
    ]
    windows = [
        window for window in model.windows
        if _accepted(window.metadata.review_status, basis)
        and window.opening_id in openings
    ]

    included = {
        *walls,
        *openings,
        *(room.id for room in rooms),
        *(door.id for door in doors),
        *(window.id for window in windows),
    }
    quantity_collections = (
        model.walls, model.openings, model.rooms, model.doors, model.windows,
    )
    all_quantity_objects = {
        item.id for collection in quantity_collections for item in collection
    }
    active_object_ids = {
        item.id for collection in quantity_collections for item in collection
        if item.metadata.review_status != ReviewStatus.REJECTED
    }
    wall_length_px = sum(wall.length_px for wall in walls.values())
    opening_width_px = sum(opening.width_px for opening in openings.values())
    floor_area_px = sum(room.area_px for room in rooms)
    room_perimeter_px = sum(room.perimeter_px for room in rooms)

    physical = {
        "wall_centerline_length": wall_length_px / scale if scale_confirmed else None,
        "opening_width": opening_width_px / scale if scale_confirmed else None,
        "floor_area": floor_area_px / (scale * scale) if scale_confirmed else None,
        "ceiling_area": floor_area_px / (scale * scale) if scale_confirmed else None,
        "room_perimeter": room_perimeter_px / scale if scale_confirmed else None,
    }
    unconfirmed_remaining = any(
        item.metadata.review_status not in {
            ReviewStatus.CONFIRMED, ReviewStatus.REJECTED,
        }
        for collection in quantity_collections
        for item in collection
    )
    dependency_exclusions = active_object_ids - included
    blocking_issues = any(
        issue.severity == IssueSeverity.ERROR
        and (
            model.id in issue.affected_object_ids
            or bool(included.intersection(issue.affected_object_ids))
        )
        for issue in model.validation_issues
    )
    complete = (
        basis == QuantityBasis.VERIFIED
        and scale_confirmed
        and not unconfirmed_remaining
        and not dependency_exclusions
        and not blocking_issues
    )
    warnings: list[str] = []
    if basis == QuantityBasis.PROVISIONAL:
        warnings.append("Provisional totals include non-rejected automatic candidates.")
    if not scale_confirmed:
        warnings.append("Calibrated quantities are unavailable until scale is confirmed.")
    if basis == QuantityBasis.VERIFIED and unconfirmed_remaining:
        warnings.append("Verified totals are partial while reviewable objects remain.")
    if dependency_exclusions:
        warnings.append("Objects with excluded or invalid dependencies are omitted.")
    if blocking_issues:
        warnings.append("Open structural validation errors prevent authoritative totals.")

    return QuantitySummary(
        model_id=model.id,
        model_revision=model.revision,
        basis=basis,
        unit=model.scale.unit,
        scale_confirmed=scale_confirmed,
        complete=complete,
        authoritative=complete,
        counts={
            "walls": len(walls),
            "openings": len(openings),
            "doors": len(doors),
            "windows": len(windows),
            "rooms": len(rooms),
        },
        pixel_measurements={
            "wall_centerline_length_px": wall_length_px,
            "opening_width_px": opening_width_px,
            "floor_area_px": floor_area_px,
            "ceiling_area_px": floor_area_px,
            "room_perimeter_px": room_perimeter_px,
        },
        calibrated_measurements=physical,
        included_object_ids=sorted(included),
        excluded_object_ids=sorted(all_quantity_objects - included),
        warnings=warnings,
    )
