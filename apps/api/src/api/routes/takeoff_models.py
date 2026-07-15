"""Read and revise persisted editable takeoff models."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.model_store import get_model_repository
from vision.domain.commands import (
    DomainCommandError,
    move_wall_endpoint,
    set_review_status,
    set_scale,
)
from vision.domain.models import Coordinate, ReviewStatus
from vision.domain.repository import (
    ModelNotFoundError,
    ModelRepository,
    RevisionConflictError,
)
from vision.domain.serialize import to_json_dict


router = APIRouter(prefix="/api/takeoff/models", tags=["takeoff-models"])


class ScaleUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    pixels_per_unit: float = Field(gt=0)
    unit: str = Field(min_length=1, max_length=16)
    actor: str = Field(default="user", min_length=1, max_length=128)


class ReviewUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    status: ReviewStatus
    locked: bool | None = None
    actor: str = Field(default="user", min_length=1, max_length=128)


class WallEndpointUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    x: float
    y: float
    actor: str = Field(default="user", min_length=1, max_length=128)


def _get(repository: ModelRepository, model_id: str):
    try:
        return repository.get(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _check_expected(model, expected_revision: int) -> None:
    if model.revision != expected_revision:
        raise HTTPException(
            status_code=409,
            detail=(
                f"model {model.id} is revision {model.revision}, "
                f"not expected revision {expected_revision}"
            ),
        )


def _save(repository: ModelRepository, model, expected_revision: int) -> None:
    try:
        repository.save(model, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{model_id}")
def get_model(
    model_id: str,
    repository: ModelRepository = Depends(get_model_repository),
):
    return {"model": to_json_dict(_get(repository, model_id))}


@router.put("/{model_id}/scale")
def update_scale(
    model_id: str,
    request: ScaleUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = set_scale(
            model, pixels_per_unit=request.pixels_per_unit,
            unit=request.unit, actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}

@router.put("/{model_id}/objects/{object_id}/review")
def update_review_status(
    model_id: str,
    object_id: str,
    request: ReviewUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = set_review_status(
            model, object_id=object_id, status=request.status,
            locked=request.locked, actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.put("/{model_id}/walls/{wall_id}/endpoints/{endpoint}")
def update_wall_endpoint(
    model_id: str,
    wall_id: str,
    endpoint: str,
    request: WallEndpointUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = move_wall_endpoint(
            model, wall_id=wall_id, endpoint=endpoint,
            point=Coordinate(request.x, request.y), actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}
