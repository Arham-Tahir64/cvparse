"""Read and revise persisted editable takeoff models."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, Field

from api.model_store import get_model_repository, get_source_asset_repository
from vision.adapters.domain_annotation_adapter import (
    to_model_annotation_document,
    to_model_svg,
)
from vision.adapters.domain_pdf import DomainRenderError, render_reviewed_pdf
from vision.domain.commands import (
    add_opening,
    add_wall,
    delete_wall,
    DomainCommandError,
    move_wall_endpoint,
    redo_last_edit,
    set_approval_status,
    set_review_status,
    set_scale,
    split_wall,
    undo_last_edit,
    update_opening_geometry,
)
from vision.domain.models import (
    ApprovalStatus,
    Coordinate,
    OpeningKind,
    ReviewStatus,
)
from vision.domain.quantities import QuantityBasis, calculate_quantities
from vision.domain.repository import (
    ModelNotFoundError,
    ModelRevisionNotFoundError,
    ModelRepository,
    RevisionConflictError,
)
from vision.domain.serialize import to_json_dict
from vision.domain.source_assets import (
    SourceAssetIntegrityError,
    SourceAssetNotFoundError,
    SourceAssetRepository,
)


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


class WallCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    thickness_px: float = Field(gt=0)
    wall_type: str = Field(default="unknown", min_length=1, max_length=64)
    snap_tolerance_px: float | None = Field(default=None, ge=0)
    actor: str = Field(default="user", min_length=1, max_length=128)


class WallDelete(BaseModel):
    expected_revision: int = Field(ge=1)
    cascade: bool = False
    actor: str = Field(default="user", min_length=1, max_length=128)


class WallSplit(BaseModel):
    expected_revision: int = Field(ge=1)
    x: float
    y: float
    projection_tolerance_px: float | None = Field(default=None, ge=0)
    actor: str = Field(default="user", min_length=1, max_length=128)


class OpeningCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    x: float
    y: float
    width_px: float = Field(gt=0)
    kind: OpeningKind = OpeningKind.UNKNOWN
    projection_tolerance_px: float | None = Field(default=None, ge=0)
    actor: str = Field(default="user", min_length=1, max_length=128)


class OpeningGeometryUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    x: float
    y: float
    width_px: float = Field(gt=0)
    projection_tolerance_px: float | None = Field(default=None, ge=0)
    actor: str = Field(default="user", min_length=1, max_length=128)


class ApprovalUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    status: ApprovalStatus
    actor: str = Field(default="user", min_length=1, max_length=128)


class HistoryUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
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


def _get_revision(repository: ModelRepository, model_id: str, revision: int):
    try:
        return repository.get_revision(model_id, revision)
    except ModelRevisionNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{model_id}")
def get_model(
    model_id: str,
    repository: ModelRepository = Depends(get_model_repository),
):
    return {"model": to_json_dict(_get(repository, model_id))}


@router.get("/{model_id}/revisions/{revision}")
def get_model_revision(
    model_id: str,
    revision: int = Path(ge=1),
    repository: ModelRepository = Depends(get_model_repository),
):
    return {"model": to_json_dict(_get_revision(repository, model_id, revision))}


@router.post("/{model_id}/undo")
def undo_model_edit(
    model_id: str,
    request: HistoryUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    if not model.undo_revision_stack:
        raise HTTPException(status_code=422, detail="nothing to undo")
    snapshot = _get_revision(repository, model_id, model.undo_revision_stack[-1])
    try:
        updated = undo_last_edit(model, snapshot, actor=request.actor)
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.post("/{model_id}/redo")
def redo_model_edit(
    model_id: str,
    request: HistoryUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    if not model.redo_revision_stack:
        raise HTTPException(status_code=422, detail="nothing to redo")
    snapshot = _get_revision(repository, model_id, model.redo_revision_stack[-1])
    try:
        updated = redo_last_edit(model, snapshot, actor=request.actor)
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.get("/{model_id}/annotations")
def get_model_annotations(
    model_id: str,
    repository: ModelRepository = Depends(get_model_repository),
):
    return {"annotations": to_model_annotation_document(_get(repository, model_id))}


@router.get("/{model_id}/overlay.svg")
def get_model_overlay(
    model_id: str,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    return Response(
        content=to_model_svg(model), media_type="image/svg+xml",
        headers={"ETag": f'"{model.id}:{model.revision}"'},
    )


@router.get("/{model_id}/quantities")
def get_model_quantities(
    model_id: str,
    basis: QuantityBasis = Query(default=QuantityBasis.PROVISIONAL),
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    return {"quantities": calculate_quantities(model, basis).to_dict()}


@router.get("/{model_id}/reviewed.pdf")
def get_reviewed_pdf(
    model_id: str,
    repository: ModelRepository = Depends(get_model_repository),
    source_repository: SourceAssetRepository = Depends(get_source_asset_repository),
):
    model = _get(repository, model_id)
    try:
        source_content = source_repository.get(model.source.fingerprint)
    except SourceAssetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SourceAssetIntegrityError as exc:
        raise HTTPException(
            status_code=500,
            detail="persisted source asset failed integrity validation",
        ) from exc
    try:
        content = render_reviewed_pdf(
            source_content, model.source.mime_type, model,
        )
    except DomainRenderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=content, media_type="application/pdf",
        headers={
            "ETag": f'"{model.id}:{model.revision}:pdf"',
            "Content-Disposition": f'inline; filename="{model.id}-r{model.revision}.pdf"',
        },
    )


@router.put("/{model_id}/approval")
def update_approval_status(
    model_id: str,
    request: ApprovalUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = set_approval_status(
            model, status=request.status, actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


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


@router.post("/{model_id}/walls")
def create_wall(
    model_id: str,
    request: WallCreate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = add_wall(
            model,
            start=Coordinate(request.start_x, request.start_y),
            end=Coordinate(request.end_x, request.end_y),
            thickness_px=request.thickness_px,
            wall_type=request.wall_type,
            snap_tolerance_px=request.snap_tolerance_px,
            actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.delete("/{model_id}/walls/{wall_id}")
def remove_wall(
    model_id: str,
    wall_id: str,
    request: WallDelete,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = delete_wall(
            model, wall_id=wall_id, cascade=request.cascade, actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.post("/{model_id}/walls/{wall_id}/split")
def split_model_wall(
    model_id: str,
    wall_id: str,
    request: WallSplit,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = split_wall(
            model,
            wall_id=wall_id,
            point=Coordinate(request.x, request.y),
            projection_tolerance_px=request.projection_tolerance_px,
            actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.post("/{model_id}/walls/{wall_id}/openings")
def create_opening(
    model_id: str,
    wall_id: str,
    request: OpeningCreate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = add_opening(
            model,
            wall_id=wall_id,
            center=Coordinate(request.x, request.y),
            width_px=request.width_px,
            kind=request.kind,
            projection_tolerance_px=request.projection_tolerance_px,
            actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}


@router.put("/{model_id}/openings/{opening_id}/geometry")
def revise_opening_geometry(
    model_id: str,
    opening_id: str,
    request: OpeningGeometryUpdate,
    repository: ModelRepository = Depends(get_model_repository),
):
    model = _get(repository, model_id)
    _check_expected(model, request.expected_revision)
    try:
        updated = update_opening_geometry(
            model,
            opening_id=opening_id,
            center=Coordinate(request.x, request.y),
            width_px=request.width_px,
            projection_tolerance_px=request.projection_tolerance_px,
            actor=request.actor,
        )
    except DomainCommandError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _save(repository, updated, request.expected_revision)
    return {"model": to_json_dict(updated)}
