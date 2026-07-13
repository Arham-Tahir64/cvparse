"""Pipeline orchestrator: runs modules 01-11 in numeric order.

Each module reads the shared PipelineState, mutates the fields it owns, and
returns it. Stage timings are recorded in state.debug.stage_timings.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np

from . import (
    door_detection,
    junction_snapping,
    line_detection,
    line_filters,
    ocr_labeling,
    preprocessing,
    room_extraction,
    structural_roi,
    wall_extraction,
    window_detection,
)
from .config import PipelineConfig
from .models import CVTakeoffResult, PipelineError, PipelineState

logger = logging.getLogger("flowbuildr.cv.pipeline")

_STAGES = [
    ("01_preprocessing", preprocessing.run),
    ("02_structural_roi", structural_roi.run),
    ("03_line_detection", line_detection.run),
    ("04_line_filters", line_filters.run),
    ("05_wall_extraction", wall_extraction.run),
    ("06_junction_snapping", junction_snapping.run),
    ("07_door_detection", door_detection.run),
    ("08_window_detection", window_detection.run),
    ("09_room_extraction", room_extraction.run),
    ("10_ocr_labeling", ocr_labeling.run),
]


def run_pipeline_state(
    image: Optional[np.ndarray] = None,
    source_path: Optional[str] = None,
    mime_type: str = "image/png",
    config: Optional[PipelineConfig] = None,
    page_number: int = 0,
    skip_stages: tuple[str, ...] = (),
    tolerate_stage_errors: bool = False,
) -> PipelineState:
    """Run the stages and return the full PipelineState.

    `skip_stages` lets debug tools omit stages by name (e.g. "10_ocr_labeling"
    when no OCR engine is installed); a skipped stage is recorded in
    state.debug.messages. `tolerate_stage_errors` (debug tools only) records a
    failed stage in state.debug.messages and continues instead of raising -
    the API path keeps the spec's fail-loud behavior.
    """
    config = config or PipelineConfig()
    state = PipelineState(
        config=config, mime_type=mime_type, source_path=source_path,
        page_number=page_number,
    )
    state.image = image

    for name, stage in _STAGES:
        if name in skip_stages:
            msg = f"stage {name} skipped by caller"
            logger.warning(msg)
            state.debug.messages.append(msg)
            continue
        start = time.perf_counter()
        try:
            state = stage(state)
        except PipelineError as exc:
            if not tolerate_stage_errors:
                raise
            msg = f"stage {name} failed: {exc}"
            logger.warning(msg)
            state.debug.messages.append(msg)
            continue
        elapsed = time.perf_counter() - start
        state.debug.stage_timings[name] = elapsed
        logger.info("%s completed in %.3fs", name, elapsed)
    return state


def run_pipeline(
    image: Optional[np.ndarray] = None,
    source_path: Optional[str] = None,
    mime_type: str = "image/png",
    config: Optional[PipelineConfig] = None,
    page_number: int = 0,
) -> CVTakeoffResult:
    """Run the full pipeline on a pre-loaded grayscale image or a file path."""
    config = config or PipelineConfig()
    state = run_pipeline_state(
        image=image, source_path=source_path, mime_type=mime_type,
        config=config, page_number=page_number,
    )
    result = state.to_takeoff_result()
    if config.generate_preview and state.image is not None:
        ok, buf = cv2.imencode(".png", state.image)
        if ok:
            result.preview_image = base64.b64encode(buf.tobytes()).decode("ascii")
    return result
