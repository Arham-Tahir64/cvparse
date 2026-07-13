"""Dedicated drafting-annotation removal before structural detection.

The first line/OCR pass supplies semantic proposals. This stage builds an
explicit removal mask, protects provisional paired wall faces, repairs only
short gaps inside those protected corridors, and replaces ``binary_masked``
with a cleaned structural binary. The pipeline then re-runs line detection and
classification on that cleaned input.
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np

from .geometry import angle_diff_rad
from .line_filters import _DIMENSION_TEXT_RE
from .models import IdGenerator, LineSegment, PipelineState, Point
from . import wall_extraction

logger = logging.getLogger("flowbuildr.cv.drafting_removal")

MODULE = "05_drafting_removal"

_REMOVAL_CLASSES = {"dimension", "grid", "leader", "hatch"}
_WALL_VETO_CLASSES = {"dimension", "grid", "leader", "hatch"}


def run(state: PipelineState) -> PipelineState:
    config = state.config
    source = state.binary_masked if state.binary_masked is not None else state.binary
    if source is None:
        return state

    if state.semantic_plan_mask is None:
        from .room_extraction import build_semantic_plan_mask
        state.semantic_plan_mask = build_semantic_plan_mask(state)

    drafting = np.zeros_like(source)
    classified = state.classified_lines
    dimension_segments = [s for s in classified if s.classification == "dimension"]

    # Location is the strongest cue: retain only ink in the semantic plan
    # envelope, which already includes a configurable exterior-wall margin.
    if state.semantic_plan_mask is not None:
        outside = cv2.bitwise_and(source, cv2.bitwise_not(state.semantic_plan_mask))
        drafting = cv2.bitwise_or(drafting, outside)

    for segment in classified:
        if segment.classification in _REMOVAL_CLASSES:
            _draw_segment(drafting, segment, config)
            if segment.classification == "dimension":
                radius = max(1, int(config.drafting_endpoint_radius_px))
                for point in (segment.start, segment.end):
                    cv2.circle(drafting, _pixel(point), radius, 255, cv2.FILLED)

    # Extension lines often have no nearby text of their own. Attach short,
    # thin perpendicular segments to already-supported dimension baselines.
    extension_ids = _extension_line_ids(dimension_segments, classified, config)
    for segment in classified:
        if segment.id in extension_ids:
            _draw_segment(drafting, segment, config)

    # Remove only measurement-like OCR boxes. Room labels remain available as
    # semantic seeds and cannot punch holes in structural walls.
    text_pad = max(0, int(config.drafting_text_pad_px))
    for text in state.raw_texts:
        if not _DIMENSION_TEXT_RE.search(text.text):
            continue
        x0, y0, x1, y1 = text.bbox
        cv2.rectangle(
            drafting,
            (max(0, int(math.floor(x0)) - text_pad),
             max(0, int(math.floor(y0)) - text_pad)),
            (min(source.shape[1] - 1, int(math.ceil(x1)) + text_pad),
             min(source.shape[0] - 1, int(math.ceil(y1)) + text_pad)),
            255, cv2.FILLED,
        )

    protection, provisional_walls = _structural_protection(state, source)

    cleaned = source.copy()
    cleaned[drafting > 0] = 0

    # Restore original wall-face pixels and close only small, axis-aligned gaps
    # inside protected corridors. Door/window openings are much larger than the
    # repair kernel and remain open.
    protected_ink = cv2.bitwise_and(source, protection)
    gap = max(1, int(config.drafting_repair_gap_px))
    horizontal = cv2.morphologyEx(
        protected_ink, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8)
    )
    vertical = cv2.morphologyEx(
        protected_ink, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8)
    )
    repaired = cv2.bitwise_and(cv2.bitwise_or(horizontal, vertical), protection)
    cleaned = cv2.bitwise_or(cleaned, repaired)

    # Location removal is final: a spurious provisional pair outside the plan
    # envelope must not restore schedules or exterior dimension strings.
    if state.semantic_plan_mask is not None:
        cleaned = cv2.bitwise_and(cleaned, state.semantic_plan_mask)

    cleaned_image = state.image.copy()
    cleaned_image[drafting > 0] = 255

    interior_drafting = drafting.copy()
    if state.semantic_plan_mask is not None:
        interior_drafting = cv2.bitwise_and(
            interior_drafting, state.semantic_plan_mask,
        )

    state.drafting_mask = drafting
    state.interior_drafting_mask = interior_drafting
    state.structural_protection_mask = protection
    state.binary_cleaned = cleaned
    state.cleaned_image = cleaned_image
    state.binary_masked = cleaned
    state.debug.segment_counts["05_drafting_pixels"] = int(np.count_nonzero(drafting))
    state.debug.segment_counts["05_interior_drafting_pixels"] = int(
        np.count_nonzero(interior_drafting)
    )
    state.debug.segment_counts["05_protected_walls"] = len(provisional_walls)

    removed = int(np.count_nonzero(source)) - int(np.count_nonzero(cleaned))
    logger.info(
        "drafting removal masked %d pixels; net removed %d; protected %d walls",
        int(np.count_nonzero(drafting)), removed, len(provisional_walls),
    )

    if config.debug_visualize and config.debug_output_dir:
        out_dir = os.path.join(config.debug_output_dir, MODULE)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "drafting_mask.png"), drafting)
        cv2.imwrite(
            os.path.join(out_dir, "interior_drafting_mask.png"),
            interior_drafting,
        )
        cv2.imwrite(os.path.join(out_dir, "structural_protection_mask.png"), protection)
        cv2.imwrite(os.path.join(out_dir, "cleaned_binary.png"), cleaned)
        cv2.imwrite(os.path.join(out_dir, "cleaned_image.png"), cleaned_image)

    return state


def _draw_segment(mask: np.ndarray, segment: LineSegment, config) -> None:
    thickness = max(
        1, min(9, int(round(segment.thickness)) + 2 * int(config.drafting_segment_pad_px))
    )
    cv2.line(mask, _pixel(segment.start), _pixel(segment.end), 255, thickness)


def _pixel(point: Point) -> tuple[int, int]:
    return int(round(point.x)), int(round(point.y))


def _extension_line_ids(
    dimensions: list[LineSegment], segments: list[LineSegment], config,
) -> set[str]:
    ids: set[str] = set()
    endpoint_limit = float(config.drafting_extension_endpoint_px)
    angle_tolerance = math.radians(config.drafting_extension_angle_tol_deg)
    for dimension in dimensions:
        for candidate in segments:
            if candidate.id == dimension.id or candidate.id in ids:
                continue
            if candidate.length > config.drafting_extension_max_length_px:
                continue
            if candidate.thickness > config.dimension_line_max_thickness_px:
                continue
            angle = angle_diff_rad(dimension.angle_rad, candidate.angle_rad)
            if abs(angle - math.pi / 2) > angle_tolerance:
                continue
            if min(
                a.distance_to(b)
                for a in (dimension.start, dimension.end)
                for b in (candidate.start, candidate.end)
            ) <= endpoint_limit:
                ids.add(candidate.id)
    return ids


def _structural_protection(state: PipelineState, source: np.ndarray):
    config = state.config
    candidates = [
        segment for segment in state.classified_lines
        if segment.classification not in _WALL_VETO_CLASSES
        and segment.length >= config.wall_min_length_px
        and _inside_mask(segment.midpoint, state.semantic_plan_mask)
    ]
    walls, _ = wall_extraction._primary_pairing(
        candidates, source, config, IdGenerator("PW")
    )
    walls = [
        wall for wall in walls
        if wall.merge_confidence >= config.drafting_wall_protect_min_confidence
    ]
    protection = np.zeros_like(source)
    pad = max(0, int(config.drafting_wall_protect_pad_px))
    for wall in walls:
        thickness = max(1, int(round(wall.thickness)) + 2 * pad)
        cv2.line(
            protection, _pixel(wall.centerline.start), _pixel(wall.centerline.end),
            255, thickness,
        )
    return protection, walls


def _inside_mask(point: Point, mask: np.ndarray | None) -> bool:
    if mask is None:
        return True
    x, y = _pixel(point)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0
