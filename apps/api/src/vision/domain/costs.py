"""Revision-bound material quantities and costs derived from a takeoff model."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .models import OpeningKind, TakeoffModel
from .quantities import QuantityBasis, calculate_quantities


MATERIAL_CODES = (
    "drywall",
    "paint",
    "insulation",
    "framing_lumber",
    "flooring",
    "ceiling",
    "baseboard",
    "doors",
    "windows",
    "glazing",
    "door_trim",
    "window_trim",
)


class MaterialEstimateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EstimateAssumptions:
    """Project assumptions expressed in the model's calibrated length unit."""

    wall_height: float
    door_height: float
    window_height: float
    wall_finish_sides: float = 2.0
    insulation_sides: float = 1.0
    stud_spacing: float = 1.333333
    plates_per_wall: float = 3.0
    extra_studs_per_opening: int = 2
    opening_trim_sides: float = 2.0
    waste_factors: dict[str, float] = field(default_factory=dict)
    unit_costs: dict[str, float] = field(default_factory=dict)
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class MaterialLineItem:
    code: str
    description: str
    quantity: float | None
    purchase_quantity: float | None
    unit: str
    waste_factor: float
    unit_cost: float | None
    extended_cost: float | None
    source_object_ids: list[str]


@dataclass(slots=True)
class MaterialEstimate:
    model_id: str
    model_revision: int
    basis: QuantityBasis
    measurement_unit: str
    currency: str
    geometry_complete: bool
    cost_complete: bool
    authoritative: bool
    priced_subtotal: float
    line_items: list[MaterialLineItem]
    included_object_ids: list[str]
    excluded_object_ids: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["basis"] = self.basis.value
        return data


def _validate(assumptions: EstimateAssumptions) -> None:
    positive = {
        "wall_height": assumptions.wall_height,
        "door_height": assumptions.door_height,
        "window_height": assumptions.window_height,
        "wall_finish_sides": assumptions.wall_finish_sides,
        "insulation_sides": assumptions.insulation_sides,
        "stud_spacing": assumptions.stud_spacing,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if assumptions.plates_per_wall < 0:
        invalid.append("plates_per_wall")
    if assumptions.extra_studs_per_opening < 0:
        invalid.append("extra_studs_per_opening")
    if assumptions.opening_trim_sides < 0:
        invalid.append("opening_trim_sides")
    if invalid:
        raise MaterialEstimateError(
            "estimate assumptions must be non-negative or positive as applicable: "
            + ", ".join(sorted(invalid))
        )
    unknown = (
        set(assumptions.waste_factors) | set(assumptions.unit_costs)
    ) - set(MATERIAL_CODES)
    if unknown:
        raise MaterialEstimateError(
            "unknown material codes: " + ", ".join(sorted(unknown))
        )
    invalid_waste = [
        code for code, value in assumptions.waste_factors.items()
        if value < 0 or value > 10
    ]
    if invalid_waste:
        raise MaterialEstimateError(
            "waste factors must be between 0 and 10: "
            + ", ".join(sorted(invalid_waste))
        )
    invalid_costs = [
        code for code, value in assumptions.unit_costs.items() if value < 0
    ]
    if invalid_costs:
        raise MaterialEstimateError(
            "unit costs cannot be negative: " + ", ".join(sorted(invalid_costs))
        )
    if not assumptions.currency.strip():
        raise MaterialEstimateError("currency is required")


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(max(0.0, value), digits)


def calculate_material_estimate(
    model: TakeoffModel,
    assumptions: EstimateAssumptions,
    basis: QuantityBasis = QuantityBasis.PROVISIONAL,
) -> MaterialEstimate:
    """Derive traceable material lines from one immutable model revision."""
    _validate(assumptions)
    quantity_summary = calculate_quantities(model, basis)
    included = set(quantity_summary.included_object_ids)
    scale = model.scale.pixels_per_unit if quantity_summary.scale_confirmed else None

    walls = [wall for wall in model.walls if wall.id in included]
    openings = [opening for opening in model.openings if opening.id in included]
    rooms = [room for room in model.rooms if room.id in included]
    doors = [door for door in model.doors if door.id in included]
    windows = [window for window in model.windows if window.id in included]
    opening_by_id = {opening.id: opening for opening in openings}

    door_openings = [
        opening_by_id[door.opening_id] for door in doors
        if door.opening_id in opening_by_id
        and opening_by_id[door.opening_id].kind == OpeningKind.DOOR
    ]
    window_openings = [
        opening_by_id[window.opening_id] for window in windows
        if window.opening_id in opening_by_id
        and opening_by_id[window.opening_id].kind == OpeningKind.WINDOW
    ]

    quantities: dict[str, float | None]
    sources: dict[str, list[str]]
    if scale is None:
        quantities = {code: None for code in MATERIAL_CODES}
        sources = {code: [] for code in MATERIAL_CODES}
    else:
        wall_length = sum(wall.length_px / scale for wall in walls)
        floor_area = sum(room.area_px / (scale * scale) for room in rooms)
        room_perimeter = sum(room.perimeter_px / scale for room in rooms)
        door_width = sum(opening.width_px / scale for opening in door_openings)
        window_width = sum(opening.width_px / scale for opening in window_openings)

        door_finish_deduction = (
            door_width * assumptions.door_height * assumptions.wall_finish_sides
        )
        window_finish_deduction = (
            window_width * assumptions.window_height * assumptions.wall_finish_sides
        )
        gross_finish_area = (
            wall_length * assumptions.wall_height * assumptions.wall_finish_sides
        )
        net_finish_area = max(
            0.0,
            gross_finish_area - door_finish_deduction - window_finish_deduction,
        )
        gross_insulation_area = (
            wall_length * assumptions.wall_height * assumptions.insulation_sides
        )
        insulation_area = max(
            0.0,
            gross_insulation_area
            - door_width * assumptions.door_height * assumptions.insulation_sides
            - window_width * assumptions.window_height * assumptions.insulation_sides,
        )
        stud_count = sum(
            max(2, math.ceil((wall.length_px / scale) / assumptions.stud_spacing) + 1)
            for wall in walls
        ) + assumptions.extra_studs_per_opening * len(door_openings + window_openings)
        framing_lumber = (
            stud_count * assumptions.wall_height
            + wall_length * assumptions.plates_per_wall
        )
        door_room_counts = {
            door.id: sum(door.id in room.door_ids for room in rooms)
            for door in doors
        }
        baseboard_deduction = sum(
            opening_by_id[door.opening_id].width_px / scale
            * max(1, door_room_counts[door.id])
            for door in doors
            if door.opening_id in opening_by_id
        )
        glazing_area = window_width * assumptions.window_height
        door_trim = sum(
            (2 * assumptions.door_height + opening.width_px / scale)
            * assumptions.opening_trim_sides
            for opening in door_openings
        )
        window_trim = sum(
            2 * (assumptions.window_height + opening.width_px / scale)
            * assumptions.opening_trim_sides
            for opening in window_openings
        )
        quantities = {
            "drywall": net_finish_area,
            "paint": net_finish_area,
            "insulation": insulation_area,
            "framing_lumber": framing_lumber,
            "flooring": floor_area,
            "ceiling": floor_area,
            "baseboard": max(0.0, room_perimeter - baseboard_deduction),
            "doors": float(len(doors)),
            "windows": float(len(windows)),
            "glazing": glazing_area,
            "door_trim": door_trim,
            "window_trim": window_trim,
        }
        wall_ids = sorted(wall.id for wall in walls)
        room_ids = sorted(room.id for room in rooms)
        door_ids = sorted(door.id for door in doors)
        window_ids = sorted(window.id for window in windows)
        opening_ids = sorted(
            opening.id for opening in door_openings + window_openings
        )
        sources = {
            "drywall": sorted(wall_ids + opening_ids),
            "paint": sorted(wall_ids + opening_ids),
            "insulation": sorted(wall_ids + opening_ids),
            "framing_lumber": sorted(wall_ids + opening_ids),
            "flooring": room_ids,
            "ceiling": room_ids,
            "baseboard": sorted(room_ids + door_ids),
            "doors": door_ids,
            "windows": window_ids,
            "glazing": sorted(window_ids + [item.id for item in window_openings]),
            "door_trim": sorted(door_ids + [item.id for item in door_openings]),
            "window_trim": sorted(
                window_ids + [item.id for item in window_openings]
            ),
        }

    area_unit = f"{model.scale.unit}^2"
    units = {
        "drywall": area_unit,
        "paint": area_unit,
        "insulation": area_unit,
        "framing_lumber": model.scale.unit,
        "flooring": area_unit,
        "ceiling": area_unit,
        "baseboard": model.scale.unit,
        "doors": "each",
        "windows": "each",
        "glazing": area_unit,
        "door_trim": model.scale.unit,
        "window_trim": model.scale.unit,
    }
    descriptions = {
        "drywall": "Net wall-board surface",
        "paint": "Net paintable wall surface",
        "insulation": "Net insulated wall cavity surface",
        "framing_lumber": "Stud and plate lineal material",
        "flooring": "Floor finish area",
        "ceiling": "Ceiling finish area",
        "baseboard": "Room trim excluding associated doors",
        "doors": "Door units",
        "windows": "Window units",
        "glazing": "Window glazing area",
        "door_trim": "Door casing length",
        "window_trim": "Window casing length",
    }
    line_items: list[MaterialLineItem] = []
    priced_subtotal = 0.0
    unpriced: list[str] = []
    for code in MATERIAL_CODES:
        quantity = _rounded(quantities[code])
        waste = assumptions.waste_factors.get(code, 0.0)
        purchase = _rounded(quantity * (1.0 + waste) if quantity is not None else None)
        unit_cost = assumptions.unit_costs.get(code)
        extended = (
            _rounded(purchase * unit_cost, 2)
            if purchase is not None and unit_cost is not None else None
        )
        if extended is not None:
            priced_subtotal += extended
        if purchase is not None and purchase > 0 and unit_cost is None:
            unpriced.append(code)
        line_items.append(MaterialLineItem(
            code=code,
            description=descriptions[code],
            quantity=quantity,
            purchase_quantity=purchase,
            unit=units[code],
            waste_factor=waste,
            unit_cost=unit_cost,
            extended_cost=extended,
            source_object_ids=sources[code],
        ))

    warnings = list(quantity_summary.warnings)
    unassociated_doors = [
        door.id for door in doors
        if not any(door.id in room.door_ids for room in rooms)
    ]
    if unassociated_doors:
        warnings.append(
            "Door-room associations are missing; baseboard was deducted once for: "
            + ", ".join(sorted(unassociated_doors)) + "."
        )
    if unpriced:
        warnings.append("Unpriced material lines: " + ", ".join(sorted(unpriced)) + ".")
    geometry_complete = quantity_summary.complete and not unassociated_doors
    cost_complete = quantity_summary.scale_confirmed and not unpriced
    authoritative = geometry_complete and cost_complete
    return MaterialEstimate(
        model_id=model.id,
        model_revision=model.revision,
        basis=basis,
        measurement_unit=model.scale.unit,
        currency=assumptions.currency.strip().upper(),
        geometry_complete=geometry_complete,
        cost_complete=cost_complete,
        authoritative=authoritative,
        priced_subtotal=round(priced_subtotal, 2),
        line_items=line_items,
        included_object_ids=quantity_summary.included_object_ids,
        excluded_object_ids=quantity_summary.excluded_object_ids,
        warnings=warnings,
    )
