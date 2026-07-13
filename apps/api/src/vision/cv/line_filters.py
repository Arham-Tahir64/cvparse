"""Module 04 - Line Filtering.

Classify every LineSegment into wall / dimension / grid / hatch / leader /
text_baseline / unknown. Rules apply in order; first match wins. Ambiguous
segments stay unknown (a leaked dimension line is cheaper than a lost wall).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import angle_diff_rad, liang_barsky_intersects, sample_line_pixels, unit_direction
from .models import LineSegment, PipelineState, Point, TextElement

logger = logging.getLogger("flowbuildr.cv.line_filters")

MODULE = "04_line_filters"

# imperial: 12'-6", 6", 3', 12 1/2", 3'-4 1/2"; metric: 1500mm, 1.5m
_DIMENSION_TEXT_RE = re.compile(
    r"""(?x)
    (
        \d+\s*'(\s*-?\s*\d+(\s+\d+/\d+)?\s*")?   # feet, optional inches
      | \d+(\s+\d+/\d+)?\s*"                       # inches only
      | \d+/\d+\s*"                                # bare fraction inches
      | \d+(\.\d+)?\s*(mm|cm|m)\b                  # metric
    )
    """
)


def run(state: PipelineState) -> PipelineState:
    config = state.config
    segments = state.raw_lines
    texts = state.raw_texts

    if state.semantic_plan_mask is None:
        # Imported lazily to keep the stage modules independently testable.
        from .room_extraction import build_semantic_plan_mask
        state.semantic_plan_mask = build_semantic_plan_mask(state)

    padded_boxes = [
        (
            t.bbox[0] - config.text_bbox_pad_px,
            t.bbox[1] - config.text_bbox_pad_px,
            t.bbox[2] + config.text_bbox_pad_px,
            t.bbox[3] + config.text_bbox_pad_px,
        )
        for t in texts
    ]
    dimension_texts = [t for t in texts if _DIMENSION_TEXT_RE.search(t.text)]

    midpoints = np.array([[s.midpoint.x, s.midpoint.y] for s in segments]) if segments else None
    kd_tree = cKDTree(midpoints) if segments else None

    classified: list[LineSegment] = []
    for i, seg in enumerate(segments):
        cls = _classify(i, seg, segments, kd_tree, padded_boxes, texts,
                        dimension_texts, state, config)
        classified.append(dataclasses.replace(seg, classification=cls))

    state.classified_lines = classified
    counts: dict[str, int] = {}
    for s in classified:
        counts[s.classification] = counts.get(s.classification, 0) + 1
    state.debug.segment_counts["04_classified"] = len(classified)
    logger.debug("classification counts: %s", counts)

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "04_classified_lines.png"),
            visualize(state, state.image),
        )
    return state


def _classify(i, seg, segments, kd_tree, padded_boxes, texts, dimension_texts,
              state, config) -> str:
    # Rule 0 - midpoint outside structural ROI
    if state.structural_roi_mask is not None:
        mid = seg.midpoint
        h, w = state.structural_roi_mask.shape[:2]
        mx, my = int(round(mid.x)), int(round(mid.y))
        if not (0 <= mx < w and 0 <= my < h) or state.structural_roi_mask[my, mx] == 0:
            return "text_baseline"

    # Rule 0.25 - geometry outside the OCR-seeded architectural envelope is
    # drafting context, regardless of unreliable LSD stroke thickness.
    if state.semantic_plan_mask is not None:
        mid = seg.midpoint
        h, w = state.semantic_plan_mask.shape[:2]
        mx, my = int(round(mid.x)), int(round(mid.y))
        if not (0 <= mx < w and 0 <= my < h) or state.semantic_plan_mask[my, mx] == 0:
            return "dimension"

    # Rule 0.5 - thin line outside the tight wall-mass (pre-dilation) core:
    # dimension strings live in the margins around the plan body
    if state.structural_core_mask is not None and (
        seg.thickness <= config.dimension_line_max_thickness_px
    ):
        mid = seg.midpoint
        h, w = state.structural_core_mask.shape[:2]
        mx, my = int(round(mid.x)), int(round(mid.y))
        if not (0 <= mx < w and 0 <= my < h) or state.structural_core_mask[my, mx] == 0:
            return "dimension"

    # Rule 1 - intersects a text bbox
    for box in padded_boxes:
        if liang_barsky_intersects(seg.start, seg.end, box):
            return "text_baseline"

    # Rule 2 - leader: short with an endpoint near a text bbox
    if seg.length < config.leader_max_length_px:
        for box in padded_boxes:
            for p in (seg.start, seg.end):
                if _point_to_rect_distance(p, box) <= config.text_proximity_px:
                    return "leader"

    # Rule 3 - dimension line
    if (
        seg.thickness <= config.dimension_line_max_thickness_px
        and seg.length >= config.dimension_min_length_px
        and (
            _has_tick_marks(seg, segments, config)
            or _has_dimension_text_nearby(seg, dimension_texts, config)
        )
    ):
        return "dimension"

    # Rule 4 - hatch
    if kd_tree is not None:
        angle_tol = math.radians(config.hatch_angle_tol_deg)

        if seg.length <= config.hatch_short_line_max_px:
            neighbor_ids = kd_tree.query_ball_point(
                [seg.midpoint.x, seg.midpoint.y], config.hatch_neighbor_radius_px
            )
            parallel_short = sum(
                1 for j in neighbor_ids
                if j != i
                and segments[j].length <= config.hatch_short_line_max_px
                and angle_diff_rad(segments[j].angle_rad, seg.angle_rad) <= angle_tol
            )
            if parallel_short >= config.hatch_cluster_density_threshold:
                return "hatch"

        # Rule 4b - diagonal hatch cluster: on Manhattan plans, a dense group
        # of parallel non-axis-aligned lines (any length) is a hatch pattern
        # (e.g. dropped-ceiling shading), not walls
        if (
            config.hatch_diagonal_cluster
            and config.manhattan
            and not seg.is_horizontal
            and not seg.is_vertical
        ):
            neighbor_ids = kd_tree.query_ball_point(
                [seg.midpoint.x, seg.midpoint.y],
                max(config.hatch_neighbor_radius_px, seg.length / 2.0),
            )
            parallel_diag = sum(
                1 for j in neighbor_ids
                if j != i
                and not segments[j].is_horizontal
                and not segments[j].is_vertical
                and angle_diff_rad(segments[j].angle_rad, seg.angle_rad) <= angle_tol
            )
            if parallel_diag >= config.hatch_cluster_density_threshold:
                return "hatch"

    # Rule 5 - dashed grid line
    if seg.length > config.grid_min_length_px and state.binary is not None:
        values = sample_line_pixels(state.binary, seg.start, seg.end)
        runs = _background_runs(values)
        if len(runs) >= config.grid_min_gap_runs:
            avg_gap = sum(runs) / len(runs)
            if avg_gap > config.grid_dash_min_gap_px:
                return "grid"

    # Rule 6 - oversized thickness
    if seg.thickness > config.wall_thickness_max_px:
        return "text_baseline"

    # Rule 7
    return "unknown"


def _point_to_rect_distance(p: Point, box) -> float:
    dx = max(box[0] - p.x, 0.0, p.x - box[2])
    dy = max(box[1] - p.y, 0.0, p.y - box[3])
    return math.hypot(dx, dy)


def _has_tick_marks(seg: LineSegment, segments: list[LineSegment], config) -> bool:
    """Both endpoints have a short roughly-perpendicular segment within reach."""
    found = {0: False, 1: False}
    for other in segments:
        if other is seg:
            continue
        if other.length > config.hatch_short_line_max_px:
            continue
        # arch tick marks are often 45-degree slashes, not strict perpendiculars
        if angle_diff_rad(other.angle_rad, seg.angle_rad) < math.radians(30):
            continue
        for idx, endpoint in enumerate((seg.start, seg.end)):
            if found[idx]:
                continue
            if min(
                endpoint.distance_to(other.start),
                endpoint.distance_to(other.end),
                endpoint.distance_to(other.midpoint),
            ) <= config.dimension_tick_search_px:
                found[idx] = True
        if found[0] and found[1]:
            return True
    return False


def _has_dimension_text_nearby(seg, dimension_texts: list[TextElement], config) -> bool:
    """Associate a dimension baseline with nearby, parallel measurement text.

    Architectural baselines are commonly much longer than their label, so
    midpoint-to-centre distance rejects the correct association near either
    end. Use distance to the complete segment instead, represented by an
    expanded OCR rectangle, and require the line direction to agree with the
    text box's major axis. The orientation check prevents nearby room labels
    from turning unrelated thin structural faces into dimensions.
    """
    distance = float(config.dimension_text_dist_px)
    angle_tolerance = math.radians(config.dimension_text_angle_tol_deg)
    for text in dimension_texts:
        x0, y0, x1, y1 = text.bbox
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        text_angle = 0.0 if width >= height else math.pi / 2
        if angle_diff_rad(seg.angle_rad, text_angle) > angle_tolerance:
            continue
        expanded = (x0 - distance, y0 - distance,
                    x1 + distance, y1 + distance)
        if liang_barsky_intersects(seg.start, seg.end, expanded):
            return True
    return False


def _background_runs(values: np.ndarray) -> list[int]:
    """Lengths of contiguous background (0) runs strictly inside the samples."""
    runs: list[int] = []
    count = 0
    started_ink = False
    for v in values:
        if v > 0:
            if started_ink and count > 0:
                runs.append(count)
            count = 0
            started_ink = True
        elif started_ink:
            count += 1
    return runs


_COLORS = {
    "unknown": (0, 0, 255),
    "dimension": (0, 255, 255),
    "grid": (255, 0, 0),
    "hatch": (160, 160, 160),
    "leader": (0, 128, 255),
    "text_baseline": (80, 80, 80),
    "wall": (0, 200, 0),
}


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for seg in state.classified_lines:
        cv2.line(
            overlay,
            (int(seg.start.x), int(seg.start.y)),
            (int(seg.end.x), int(seg.end.y)),
            _COLORS.get(seg.classification, (0, 0, 255)), 1,
        )
    return overlay
