"""POST /api/cv/takeoff - run the CV pipeline on an uploaded plan."""
from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.adapters.annotation_adapter import to_annotation_document
from vision.cv import serialize
from vision.cv.config import PipelineConfig
from vision.cv.models import PipelineError
from vision.cv.pipeline import run_pipeline
from vision.cv.preprocessing import load_image
from vision.domain.import_cv import import_cv_result
from vision.domain.serialize import to_json_dict as domain_to_json_dict

logger = logging.getLogger("flowbuildr.api.cv_takeoff")

router = APIRouter(prefix="/api/cv", tags=["cv"])


@router.post("/takeoff")
async def cv_takeoff(
    file: UploadFile = File(...),
    mime_type: str = Form(None),
    page_number: int = Form(0),
    include_annotations: bool = Form(False),
    include_model: bool = Form(False),
):
    mime = mime_type or file.content_type or "application/octet-stream"
    file_bytes = await file.read()
    config = PipelineConfig()

    try:
        image = load_image(file_bytes, mime, config.working_dpi, page_number)
        result = run_pipeline(
            image=image, mime_type=mime, config=config, page_number=page_number
        )
    except PipelineError as exc:
        logger.warning("pipeline failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = {
        "takeoff": serialize.to_json_dict(result),
        "annotations": to_annotation_document(result) if include_annotations else None,
    }
    if include_model:
        fingerprint = hashlib.sha256(file_bytes).hexdigest()
        model = import_cv_result(
            result, source_fingerprint=fingerprint, mime_type=mime,
        )
        response["model"] = domain_to_json_dict(model)
    return response
