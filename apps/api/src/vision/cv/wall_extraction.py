"""Module 05 - Wall Extraction.

Primary parallel-face pairing with pixel-level face support, thin-wall
recovery, single-face fallback, and visual thickness measurement. Every wall
carries full provenance.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os

import cv2
import numpy as np

from .geometry import angle_diff_rad, point_to_line_distance, segment_orientation
from .models import (
    IdGenerator, LineSegment, NoWallsFoundError, PipelineState, Point, Wall,
)

logger = logging.getLogger("flowbuildr.cv.wall_extraction")

MODULE = "05_wall_extraction"


def run(state: PipelineState) -> PipelineState:
    config = state.config
    id_gen = IdGenerator("W")
    binary = (state.binary_cleaned if state.binary_cleaned is not None
              else state.binary_masked if state.binary_masked is not None
              else state.binary)

    candidates = [
        s for s in state.classified_lines
        if s.classification in ("unknown", "wall") and s.length >= config.wall_min_length_px
    ]
    walls, used_ids = _primary_pairing(candidates, binary, config, id_gen)
    logger.debug("primary pairing produced %d walls", len(walls))

    unused = [
        s for s in state.classified_lines
        if s.classification == "unknown" and s.id not in used_ids
    ]
    thin_walls, thin_used = _thin_wall_recovery(unused, walls, state, config, id_gen)
    walls.extend(thin_walls)
    used_ids |= thin_used
    logger.debug("thin-wall recovery produced %d walls", len(thin_walls))

    remaining = [s for s in unused if s.id not in used_ids]
    single_walls = _single_face_fallback(remaining, config, id_gen)
    walls.extend(single_walls)
    logger.debug("single-face fallback produced %d walls", len(single_walls))

    for wall in walls:
        wall.visual_thickness = _measure_visual_thickness(wall, binary, config)

    if not walls:
        raise NoWallsFoundError(MODULE, "no walls found by any extraction branch")

    state.walls = walls
    state.debug.segment_counts["05_walls"] = len(walls)

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "05_walls.png"),
            visualize(state, state.image),
        )
    return state


# ---------------------------------------------------------------------------
# Step 1 - primary parallel face pairing
# ---------------------------------------------------------------------------

def _primary_pairing(candidates, binary, config, id_gen):
    angle_tol = math.radians(config.parallel_angle_tolerance_deg)
    n_bins = max(1, int(round(math.pi / angle_tol)))
    buckets: dict[int, list[int]] = {}
    for i, seg in enumerate(candidates):
        b = int(seg.angle_rad / math.pi * n_bins) % n_bins
        for bb in (b - 1, b, b + 1):
            buckets.setdefault(bb % n_bins, []).append(i)

    scored: list[tuple[float, int, int, dict]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        for ai in range(len(bucket)):
            for bi in range(ai + 1, len(bucket)):
                i, j = sorted((bucket[ai], bucket[bi]))
                if i == j or (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                info = _validate_pair(candidates[i], candidates[j], binary, config)
                if info is not None:
                    scored.append((info["merge_confidence"], i, j, info))

    scored.sort(key=lambda t: t[0], reverse=True)
    walls: list[Wall] = []
    used_ids: set[str] = set()
    used_idx: set[int] = set()
    for confidence, i, j, info in scored:
        if i in used_idx or j in used_idx:
            continue
        wall = _build_paired_wall(candidates[i], candidates[j], info, config, id_gen)
        if wall is None:
            continue
        used_idx |= {i, j}
        used_ids |= {candidates[i].id, candidates[j].id}
        walls.append(wall)
    return walls, used_ids


def _validate_pair(a: LineSegment, b: LineSegment, binary, config):
    if angle_diff_rad(a.angle_rad, b.angle_rad) >= math.radians(
        config.parallel_angle_tolerance_deg
    ):
        return None

    dist = point_to_line_distance(b.midpoint, a.start, a.end)
    if not (config.wall_thickness_min_px <= dist <= config.wall_thickness_max_px):
        return None

    # overlap along the shared direction (use a's direction as reference)
    ux, uy = _unit(a)
    a_lo, a_hi = sorted((_proj(a.start, ux, uy), _proj(a.end, ux, uy)))
    b_lo, b_hi = sorted((_proj(b.start, ux, uy), _proj(b.end, ux, uy)))
    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
    overlap = hi - lo
    if overlap <= 0:
        return None
    shorter = min(a.length, b.length)
    overlap_ratio = overlap / shorter if shorter > 0 else 0.0
    if overlap_ratio < config.parallel_overlap_min_ratio:
        return None
    if overlap < config.parallel_overlap_min_px:
        return None

    support = _face_support(a, b, lo, hi, (ux, uy), dist, binary, config)
    if support is None:
        return None
    fit_support_ratio, _ = support

    return {
        "distance": dist,
        "interval": (lo, hi),
        "direction": (ux, uy),
        "overlap_ratio": overlap_ratio,
        "fit_support_ratio": fit_support_ratio,
        "merge_confidence": overlap_ratio * fit_support_ratio,
    }


def _face_support(a, b, lo, hi, direction, dist, binary, config):
    """Sample both faces along the overlap; return (support_ratio, run) or None."""
    ux, uy = direction
    nx, ny = -uy, ux
    a_off = _perp_offset(a.midpoint, nx, ny)
    b_off = _perp_offset(b.midpoint, nx, ny)

    step = config.face_support_sample_step_px
    pad = config.face_support_window_pad_px
    positions = np.arange(lo, hi + 1e-6, step)
    if len(positions) < 2:
        positions = np.array([lo, hi])

    h, w = binary.shape[:2]
    supported = []
    for t in positions:
        both = True
        for off in (a_off, b_off):
            hit = False
            for d in range(-pad, pad + 1):
                x = int(round(t * ux + (off + d) * nx))
                y = int(round(t * uy + (off + d) * ny))
                if 0 <= x < w and 0 <= y < h and binary[y, x] > 0:
                    hit = True
                    break
            if not hit:
                both = False
                break
        supported.append(both)

    ratio = float(np.mean(supported))
    # longest contiguous supported run, in pixels
    best_run = run = 0
    for s in supported:
        run = run + 1 if s else 0
        best_run = max(best_run, run)
    run_px = best_run * step if best_run > 1 else (step if best_run == 1 else 0)
    if run_px < config.face_support_min_run_px:
        return None
    return ratio, run_px


def _build_paired_wall(a, b, info, config, id_gen):
    lo, hi = info["interval"]
    if lo >= hi:
        return None
    ux, uy = info["direction"]
    nx, ny = -uy, ux
    mid_off = (_perp_offset(a.midpoint, nx, ny) + _perp_offset(b.midpoint, nx, ny)) / 2.0

    start = Point(lo * ux + mid_off * nx, lo * uy + mid_off * ny)
    end = Point(hi * ux + mid_off * nx, hi * uy + mid_off * ny)
    centerline = LineSegment(start, end, thickness=info["distance"])

    if config.manhattan:
        centerline = _manhattan_snap(centerline, config)

    return Wall(
        id=id_gen(),
        orientation=segment_orientation(centerline),
        centerline=centerline,
        thickness=info["distance"],
        merge_kind="paired_faces",
        fit_support_ratio=info["fit_support_ratio"],
        merge_confidence=info["merge_confidence"],
        source_ids=[a.id, b.id],
        length_px=centerline.length,
    )


def _manhattan_snap(centerline: LineSegment, config) -> LineSegment:
    angle_deg = math.degrees(centerline.angle_rad)
    snap_tol = config.manhattan_snap_angle_deg
    mid = centerline.midpoint
    half = centerline.length / 2.0
    if angle_deg <= snap_tol or angle_deg >= 180 - snap_tol:
        return LineSegment(
            Point(mid.x - half, mid.y), Point(mid.x + half, mid.y),
            thickness=centerline.thickness,
        )
    if abs(angle_deg - 90) <= snap_tol:
        return LineSegment(
            Point(mid.x, mid.y - half), Point(mid.x, mid.y + half),
            thickness=centerline.thickness,
        )
    return centerline


# ---------------------------------------------------------------------------
# Step 2 - thin-wall recovery
# ---------------------------------------------------------------------------

def _thin_wall_recovery(unused, primary_walls, state, config, id_gen):
    walls: list[Wall] = []
    used: set[str] = set()
    mask = state.structural_roi_mask
    for seg in unused:
        if not (
            config.thin_branch_min_length_px <= seg.length <= config.thin_branch_max_length_px
        ):
            continue
        if seg.thickness < config.thin_branch_min_thickness_px:
            continue

        if mask is not None:
            mid = seg.midpoint
            h, w = mask.shape[:2]
            mx, my = int(round(mid.x)), int(round(mid.y))
            if not (0 <= mx < w and 0 <= my < h) or mask[my, mx] == 0:
                continue

        needs_orthogonal = seg.length >= config.thin_branch_stub_bypass_length_px
        if needs_orthogonal and not _has_orthogonal_support(seg, primary_walls, config):
            continue

        if _overlaps_accepted_wall(seg, primary_walls + walls, config):
            continue

        walls.append(_single_segment_wall(seg, "single_face", config, id_gen))
        used.add(seg.id)
    return walls, used


def _has_orthogonal_support(seg, primary_walls, config) -> bool:
    mid = seg.midpoint
    for wall in primary_walls:
        cl = wall.centerline
        near = min(
            mid.distance_to(cl.start), mid.distance_to(cl.end),
            point_to_line_distance(mid, cl.start, cl.end),
        )
        if near > config.thin_branch_orthogonal_support_dist_px:
            continue
        if angle_diff_rad(cl.angle_rad, seg.angle_rad) > math.radians(
            90 - 2 * config.parallel_angle_tolerance_deg
        ):
            return True
    return False


def _overlaps_accepted_wall(seg, walls, config) -> bool:
    ux, uy = _unit(seg)
    s_lo, s_hi = sorted((_proj(seg.start, ux, uy), _proj(seg.end, ux, uy)))
    for wall in walls:
        cl = wall.centerline
        if angle_diff_rad(cl.angle_rad, seg.angle_rad) > math.radians(
            2 * config.parallel_angle_tolerance_deg
        ):
            continue
        if point_to_line_distance(cl.midpoint, seg.start, seg.end) > config.wall_thickness_max_px:
            continue
        w_lo, w_hi = sorted((_proj(cl.start, ux, uy), _proj(cl.end, ux, uy)))
        overlap = min(s_hi, w_hi) - max(s_lo, w_lo)
        if overlap > config.thin_branch_max_overlap_ratio * seg.length:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 3 - single-face fallback
# ---------------------------------------------------------------------------

def _single_face_fallback(remaining, config, id_gen):
    walls = []
    for seg in remaining:
        if seg.thickness >= config.single_face_min_thickness_px and (
            seg.length >= config.wall_min_length_px
        ):
            walls.append(_single_segment_wall(seg, "single_face", config, id_gen))
    return walls


def _single_segment_wall(seg: LineSegment, merge_kind, config, id_gen) -> Wall:
    centerline = LineSegment(seg.start, seg.end, thickness=seg.thickness)
    if config.manhattan:
        centerline = _manhattan_snap(centerline, config)
    return Wall(
        id=id_gen(),
        orientation=segment_orientation(centerline),
        centerline=centerline,
        thickness=seg.thickness,
        merge_kind=merge_kind,
        fit_support_ratio=1.0,
        merge_confidence=1.0,
        source_ids=[seg.id],
        length_px=centerline.length,
    )


# ---------------------------------------------------------------------------
# Step 4 - visual thickness
# ---------------------------------------------------------------------------

def _measure_visual_thickness(wall: Wall, binary, config) -> float:
    cl = wall.centerline
    length = cl.length
    margin = config.visual_thickness_endpoint_margin_px
    step = config.face_support_sample_step_px
    if length <= 2 * margin + step:
        logger.warning(
            "wall %s too short for visual thickness sampling; using thickness", wall.id
        )
        return min(wall.thickness, config.visual_thickness_max_px)

    ux, uy = _unit(cl)
    nx, ny = -uy, ux
    h, w = binary.shape[:2]
    search = config.visual_thickness_search_px
    widths = []
    for t in np.arange(margin, length - margin + 1e-6, step):
        cx = cl.start.x + t * ux
        cy = cl.start.y + t * uy
        # measure the contiguous ink blob width crossing the centerline
        pos = 0
        for d in range(0, search + 1):
            x, y = int(round(cx + d * nx)), int(round(cy + d * ny))
            if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
                break
            pos = d + 1
        neg = 0
        for d in range(1, search + 1):
            x, y = int(round(cx - d * nx)), int(round(cy - d * ny))
            if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
                break
            neg = d
        width = pos + neg
        if width > 0:
            widths.append(width)

    if not widths:
        logger.warning("wall %s has no visual thickness samples; using thickness", wall.id)
        return min(wall.thickness, config.visual_thickness_max_px)
    return float(min(np.median(widths), config.visual_thickness_max_px))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _unit(seg: LineSegment) -> tuple[float, float]:
    dx, dy = seg.end.x - seg.start.x, seg.end.y - seg.start.y
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _proj(p: Point, ux: float, uy: float) -> float:
    return p.x * ux + p.y * uy


def _perp_offset(p: Point, nx: float, ny: float) -> float:
    return p.x * nx + p.y * ny


_KIND_COLORS = {"paired_faces": (0, 200, 0), "single_face": (0, 165, 255)}


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for wall in state.walls:
        cl = wall.centerline
        color = _KIND_COLORS.get(wall.merge_kind, (0, 0, 255))
        cv2.line(overlay, (int(cl.start.x), int(cl.start.y)),
                 (int(cl.end.x), int(cl.end.y)), color, 2)
        cv2.putText(
            overlay, f"{wall.merge_confidence:.2f}",
            (int(cl.midpoint.x), int(cl.midpoint.y)),
            cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1,
        )
    return overlay
