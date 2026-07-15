"""Revisioned human correction commands and targeted recomputation."""
from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone

from .models import (
    ApprovalStatus,
    EditEvent,
    ObjectMetadata,
    ReviewStatus,
    ScaleMethod,
    TakeoffModel,
)
from .validation import validate_model


class DomainCommandError(ValueError):
    pass


def _event(
    model: TakeoffModel,
    action: str,
    actor: str,
    revision_before: int,
    affected: list[str],
    payload: dict,
) -> EditEvent:
    return EditEvent(
        id=f"edit_{uuid.uuid4().hex}", action=action, actor=actor,
        revision_before=revision_before, revision_after=model.revision,
        affected_object_ids=sorted(set(affected)), payload=payload,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _ensure_editable(model: TakeoffModel) -> None:
    if model.approval_status == ApprovalStatus.APPROVED:
        raise DomainCommandError(
            "approved model revisions must be reopened explicitly before editing"
        )


def set_scale(
    model: TakeoffModel,
    *,
    pixels_per_unit: float,
    unit: str,
    actor: str = "user",
) -> TakeoffModel:
    """Confirm scale and recompute only calibrated measurements."""
    _ensure_editable(model)
    if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 0:
        raise DomainCommandError("pixels_per_unit must be a positive finite number")
    if not unit.strip():
        raise DomainCommandError("unit is required")

    updated = copy.deepcopy(model)
    before = updated.revision
    previous = {
        "pixels_per_unit": updated.scale.pixels_per_unit,
        "unit": updated.scale.unit,
        "method": updated.scale.method.value,
        "review_status": updated.scale.review_status.value,
    }
    updated.scale.pixels_per_unit = float(pixels_per_unit)
    updated.scale.unit = unit.strip()
    updated.scale.method = ScaleMethod.MANUAL
    updated.scale.confidence = 1.0
    updated.scale.review_status = ReviewStatus.CONFIRMED

    for wall in updated.walls:
        wall.length = wall.length_px / pixels_per_unit
    for opening in updated.openings:
        opening.width = opening.width_px / pixels_per_unit
    area_scale = pixels_per_unit * pixels_per_unit
    for room in updated.rooms:
        room.area = room.area_px / area_scale
        room.perimeter = room.perimeter_px / pixels_per_unit

    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_scale", actor, before, [updated.id],
        {
            "before": previous,
            "after": {
                "pixels_per_unit": updated.scale.pixels_per_unit,
                "unit": updated.scale.unit,
                "method": updated.scale.method.value,
                "review_status": updated.scale.review_status.value,
            },
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated

def _editable_metadata(model: TakeoffModel, object_id: str) -> ObjectMetadata:
    for collection in (
        model.nodes, model.walls, model.openings,
        model.doors, model.windows, model.rooms,
    ):
        for item in collection:
            if item.id == object_id:
                return item.metadata
    raise DomainCommandError(f"object {object_id} was not found")


def set_review_status(
    model: TakeoffModel,
    *,
    object_id: str,
    status: ReviewStatus,
    locked: bool | None = None,
    actor: str = "user",
) -> TakeoffModel:
    """Confirm, reject, or reopen one object without rerunning extraction."""
    _ensure_editable(model)
    updated = copy.deepcopy(model)
    metadata = _editable_metadata(updated, object_id)
    before = updated.revision
    previous_status = metadata.review_status
    previous_locked = metadata.locked
    metadata.review_status = status
    metadata.locked = (
        status in {ReviewStatus.CONFIRMED, ReviewStatus.REJECTED}
        if locked is None else bool(locked)
    )
    metadata.revision += 1
    updated.revision += 1
    updated.edit_history.append(_event(
        updated, "set_review_status", actor, before, [object_id],
        {
            "before": {
                "review_status": previous_status.value,
                "locked": previous_locked,
            },
            "after": {
                "review_status": status.value,
                "locked": metadata.locked,
            },
        },
    ))
    updated.validation_issues = validate_model(updated)
    return updated
