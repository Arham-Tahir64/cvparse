"""Module 07 - Door Detection.

Detect doors from quarter-circle swing arcs, associate each with a wall,
record hinge and swing direction, split the parent wall at the hinge, and
record a Gap so module 08 does not re-detect the opening as a window.
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np

from .geometry import point_at_param, project_param
from .junction_snapping import split_wall_at
from .models import Door, Gap, IdGenerator, Junction, PipelineState, Point, Wall

logger = logging.getLogger("flowbuildr.cv.door_detection")

MODULE = "07_door_detection"

_ARC_SAMPLES = 72  # points sampled around each candidate circle


def run(state: PipelineState) -> PipelineState:
    config = state.config
    binary = (state.binary_cleaned if state.binary_cleaned is not None
              else state.binary_masked if state.binary_masked is not None
              else state.binary)
    # Structural cleanup intentionally removes thin annotation-like strokes,
    # which can include a valid door leaf. Use the original in-plan raster only
    # for leaf evidence; wall continuation and opening checks stay on the
    # cleaned structural binary so dimensions cannot manufacture a doorway.
    leaf_binary = state.binary if state.binary is not None else binary
    if state.semantic_plan_mask is not None:
        leaf_binary = cv2.bitwise_and(leaf_binary, state.semantic_plan_mask)

    erased = _erase_walls(binary, state.walls, config)
    circles = _hough_circles(erased, config)
    logger.debug("hough circle candidates: %d", len(circles))

    door_id_gen = IdGenerator("D")
    gap_id_gen = IdGenerator("G", start=len(state.gaps) + 1)
    wall_id_gen = IdGenerator("W", start=_max_wall_id(state.walls) + 1)
    junction_id_gen = IdGenerator("J", start=len(state.junctions) + 1)

    candidates = []
    for cx, cy, radius in circles:
        coverage, start_angle, end_angle = _arc_coverage(erased, cx, cy, radius)
        if not (config.arc_coverage_min <= coverage <= config.arc_coverage_max):
            continue
        if (state.semantic_plan_mask is not None and
                not _mask_contains(state.semantic_plan_mask, cx, cy)):
            continue
        best = None
        for wall in _nearby_walls(Point(cx, cy), state.walls, config):
            geometry = _candidate_geometry(
                binary, wall, cx, cy, radius, start_angle, end_angle, config,
                leaf_binary=leaf_binary,
            )
            if geometry is not None and (best is None or geometry[0] > best[0]):
                best = (*geometry, wall)
        if best is None:
            continue
        score, hinge, swing_end, swing_arc, wall = best
        candidates.append(
            (score, radius, coverage, cx, cy, wall, hinge, swing_end, swing_arc)
        )

    # Dedup: strongest door evidence wins; hinge-near duplicates collapse.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    accepted: list[tuple] = []
    doors: list[Door] = []
    for score, radius, coverage, cx, cy, wall, hinge, swing_end, swing_arc in candidates:
        cl = wall.centerline
        t = min(1.0, max(0.0, project_param(hinge, cl.start, cl.end)))
        if any(hinge.distance_to(d.position) <= config.door_dedup_dist_px for d in doors):
            continue

        swing = _swing_direction(cl, hinge, swing_end)
        door = Door(
            id=door_id_gen(), position=hinge, swing_end=swing_end, radius=float(radius),
            wall_id=wall.id, swing_direction=swing, swing_arc=swing_arc,
            confidence=float(score),
        )
        doors.append(door)

        fill = _arc_interior_fill(binary, cx, cy, radius)
        state.gaps.append(Gap(
            id=gap_id_gen(), wall_id=wall.id,
            orientation="H" if wall.orientation != "V" else "V",
            center=hinge, width_px=float(radius),
            bbox=(hinge.x - radius, hinge.y - radius, hinge.x + radius, hinge.y + radius),
            kind="door", wall_break_score=float(coverage), opening_fill_ratio=fill,
        ))
        accepted.append((door, wall, t))

    for door, wall, t in accepted:
        _split_wall_at_door(state, door, wall, t, wall_id_gen, junction_id_gen)

    state.doors = doors
    state.debug.segment_counts["07_doors"] = len(doors)
    logger.info("detected %d doors", len(doors))

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "07_doors.png"),
            visualize(state, state.image),
        )
    return state


def _max_wall_id(walls) -> int:
    best = 0
    for w in walls:
        try:
            best = max(best, int(w.id[1:]))
        except ValueError:
            pass
    return best


def _erase_walls(binary: np.ndarray, walls, config) -> np.ndarray:
    erased = binary.copy()
    for wall in walls:
        cl = wall.centerline
        cv2.line(
            erased,
            (int(round(cl.start.x)), int(round(cl.start.y))),
            (int(round(cl.end.x)), int(round(cl.end.y))),
            0, max(1, int(round(wall.thickness + config.wall_erase_extra_px))),
        )
    return erased


def _hough_circles(erased: np.ndarray, config) -> list[tuple[float, float, float]]:
    blurred = cv2.GaussianBlur(erased, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=config.hough_circles_dp, minDist=config.hough_circles_min_dist,
        param1=config.hough_circles_param1, param2=config.hough_circles_param2,
        minRadius=int(config.door_arc_min_radius_px),
        maxRadius=int(config.door_arc_max_radius_px),
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def _arc_coverage(image: np.ndarray, cx: float, cy: float, radius: float):
    """Longest contiguous ink run around the circumference (with wraparound).

    Returns ``(coverage_fraction, start_angle, end_angle)`` for the longest
    run. ``end_angle`` can exceed 2*pi when the run crosses angle zero; this
    preserves a continuous ordered arc for exporters.
    """
    h, w = image.shape[:2]
    angles = np.linspace(0, 2 * math.pi, _ARC_SAMPLES, endpoint=False)
    on = []
    radial_tolerance = max(2, min(6, int(round(radius * 0.04))))
    for a in angles:
        hit = False
        # Hough radii on thin anti-aliased PDF arcs routinely differ by a few
        # pixels; use a scale-aware annulus instead of a fixed one-pixel ring.
        for dr in range(-radial_tolerance, radial_tolerance + 1):
            x = int(round(cx + (radius + dr) * math.cos(a)))
            y = int(round(cy + (radius + dr) * math.sin(a)))
            if 0 <= x < w and 0 <= y < h and image[y, x] > 0:
                hit = True
                break
        on.append(hit)

    if not any(on):
        return 0.0, 0.0, 0.0
    if all(on):
        return 1.0, 0.0, 2 * math.pi

    # longest run with wraparound: double the array
    doubled = on + on
    best_len, best_start = 0, 0
    run, run_start = 0, 0
    for i, v in enumerate(doubled):
        if v:
            if run == 0:
                run_start = i
            run += 1
            if run > best_len and run_start < _ARC_SAMPLES:
                best_len, best_start = run, run_start
        else:
            run = 0
    best_len = min(best_len, _ARC_SAMPLES)
    step = 2 * math.pi / _ARC_SAMPLES
    start_angle = (best_start % _ARC_SAMPLES) * step
    end_angle = start_angle + max(0, best_len - 1) * step
    return best_len / _ARC_SAMPLES, float(start_angle), float(end_angle)


def _mask_contains(mask: np.ndarray, x: float, y: float) -> bool:
    ix, iy = int(round(x)), int(round(y))
    return 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix] > 0


def _nearby_walls(center: Point, walls, config) -> list[Wall]:
    nearby = []
    for wall in walls:
        cl = wall.centerline
        t = min(1.0, max(0.0, project_param(center, cl.start, cl.end)))
        distance = center.distance_to(point_at_param(cl.start, cl.end, t))
        if distance <= config.door_wall_snap_px + wall.thickness / 2.0:
            nearby.append((distance, wall))
    nearby.sort(key=lambda item: item[0])
    return [wall for _, wall in nearby[:6]]


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _candidate_geometry(
    binary: np.ndarray,
    wall: Wall,
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    config,
    leaf_binary: np.ndarray | None = None,
):
    """Validate a circle proposal using wall-opening and leaf evidence.

    A door arc has one endpoint parallel to the supporting wall (the jamb)
    and the other perpendicular (the open leaf). Structural wall ink should
    continue behind the hinge but disappear through the opening.
    """
    cl = wall.centerline
    dx, dy = cl.end.x - cl.start.x, cl.end.y - cl.start.y
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux, uy = dx / length, dy / length
    axis_angle = math.atan2(uy, ux)
    endpoints = (start_angle, end_angle)
    assignments = []
    for closed_index in (0, 1):
        closed_angle = endpoints[closed_index]
        leaf_angle = endpoints[1 - closed_index]
        parallel_error = min(
            _angle_error(closed_angle, axis_angle),
            _angle_error(closed_angle, axis_angle + math.pi),
        )
        perpendicular_error = min(
            _angle_error(leaf_angle, axis_angle + math.pi / 2),
            _angle_error(leaf_angle, axis_angle - math.pi / 2),
        )
        assignments.append((parallel_error + perpendicular_error,
                            parallel_error, perpendicular_error,
                            closed_angle, leaf_angle))
    _, parallel_error, perpendicular_error, closed_angle, leaf_angle = min(assignments)
    tolerance = math.radians(config.door_axis_angle_tol_deg)
    if parallel_error > tolerance or perpendicular_error > tolerance:
        return None

    # Hough proposals and wall erasure commonly truncate 10-30 degrees from a
    # thin arc. Once both endpoints agree with a rectilinear wall, snap them to
    # the architectural parallel/perpendicular axes instead of exporting the
    # truncated proposal angles.
    closed_angle = min(
        (axis_angle, axis_angle + math.pi),
        key=lambda angle: _angle_error(closed_angle, angle),
    )
    leaf_angle = min(
        (axis_angle + math.pi / 2, axis_angle - math.pi / 2),
        key=lambda angle: _angle_error(leaf_angle, angle),
    )

    t = min(1.0, max(0.0, project_param(Point(cx, cy), cl.start, cl.end)))
    hinge = point_at_param(cl.start, cl.end, t)
    if not _hinge_center_offset_valid(
        Point(cx, cy), hinge, radius, config.door_max_hinge_offset_ratio,
    ):
        return None
    opening_sign = 1.0 if (math.cos(closed_angle) * ux +
                           math.sin(closed_angle) * uy) >= 0 else -1.0
    inset = max(5.0, wall.thickness * 0.7)
    continuation = _structural_wall_support(
        binary, hinge, ux, uy, -opening_sign,
        inset, max(inset + 8.0, min(radius * 0.45, 55.0)), wall.thickness,
    )
    opening = _structural_wall_support(
        binary, hinge, ux, uy, opening_sign,
        inset, max(inset + 8.0, radius * 0.78), wall.thickness,
    )
    leaf_support = _radial_ink_support(
        leaf_binary if leaf_binary is not None else binary,
        cx, cy, radius, leaf_angle,
    )
    if continuation < config.door_min_wall_continuation:
        return None
    if opening > config.door_max_opening_support:
        return None
    if leaf_support < config.door_min_leaf_support:
        return None

    arc_delta = (leaf_angle - closed_angle + math.pi) % (2 * math.pi) - math.pi
    angles = np.linspace(closed_angle, closed_angle + arc_delta, 18)
    swing_arc = [Point(
        hinge.x + radius * math.cos(float(angle)),
        hinge.y + radius * math.sin(float(angle)),
    ) for angle in angles]
    swing_end = Point(
        hinge.x + radius * math.cos(leaf_angle),
        hinge.y + radius * math.sin(leaf_angle),
    )
    alignment = 1.0 - (parallel_error + perpendicular_error) / (2 * tolerance)
    score = (0.30 * continuation + 0.30 * (1.0 - opening) +
             0.25 * leaf_support + 0.15 * max(0.0, alignment))
    return score, hinge, swing_end, swing_arc


def _hinge_center_offset_valid(
    circle_center: Point, hinge: Point, radius: float, max_ratio: float,
) -> bool:
    """Return whether a swing-circle centre plausibly represents its hinge."""
    return radius > 0 and circle_center.distance_to(hinge) <= radius * max_ratio


def _structural_wall_support(
    binary: np.ndarray,
    hinge: Point,
    ux: float,
    uy: float,
    sign: float,
    start: float,
    end: float,
    thickness: float,
) -> float:
    """Fraction of axis samples having a wall-width cross-section.

    A thin door leaf or dimension line can put ink in the opening, but unlike a
    wall face pair it does not span a meaningful fraction of wall thickness.
    """
    if end <= start:
        return 0.0
    h, w = binary.shape[:2]
    half = max(3, int(round(thickness * 0.65)))
    min_span = max(3.0, min(8.0, thickness * 0.35))
    supports = []
    for offset in np.linspace(start, end, max(8, int(end - start) + 1)):
        x = hinge.x + sign * offset * ux
        y = hinge.y + sign * offset * uy
        ink_offsets = []
        for cross in range(-half, half + 1):
            ix = int(round(x - cross * uy))
            iy = int(round(y + cross * ux))
            if 0 <= ix < w and 0 <= iy < h and binary[iy, ix] > 0:
                ink_offsets.append(cross)
        supports.append(bool(ink_offsets) and
                        max(ink_offsets) - min(ink_offsets) >= min_span)
    return float(np.mean(supports)) if supports else 0.0


def _radial_ink_support(
    binary: np.ndarray, cx: float, cy: float, radius: float, angle: float,
) -> float:
    h, w = binary.shape[:2]
    hits = []
    for distance in np.linspace(radius * 0.15, radius * 0.88, 40):
        x = int(round(cx + distance * math.cos(angle)))
        y = int(round(cy + distance * math.sin(angle)))
        hit = False
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if 0 <= x + ox < w and 0 <= y + oy < h and binary[y + oy, x + ox] > 0:
                    hit = True
                    break
            if hit:
                break
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def _swing_direction(cl, hinge: Point, swing_end: Point) -> str:
    wx, wy = cl.end.x - cl.start.x, cl.end.y - cl.start.y
    sx, sy = swing_end.x - hinge.x, swing_end.y - hinge.y
    cross = wx * sy - wy * sx
    return "cw" if cross >= 0 else "ccw"


def _arc_interior_fill(binary: np.ndarray, cx, cy, radius) -> float:
    h, w = binary.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    if not inside.any():
        return 0.0
    region = binary[y0:y1, x0:x1]
    return 1.0 - float((region[inside] > 0).mean())


def _split_wall_at_door(state, door, wall, t, wall_id_gen, junction_id_gen):
    config = state.config
    if wall not in state.walls:  # already replaced by an earlier split
        wall = next((w for w in state.walls if w.id == door.wall_id), None)
        if wall is None:
            return
        cl = wall.centerline
        t = min(1.0, max(0.0, project_param(door.position, cl.start, cl.end)))
    if not (config.door_split_t_min < t < config.door_split_t_max):
        return
    child_a, child_b = split_wall_at(wall, door.position, wall_id_gen)
    idx = state.walls.index(wall)
    state.walls[idx: idx + 1] = [child_a, child_b]

    junction = Junction(
        id=junction_id_gen(), point=door.position,
        walls=[child_a.id, child_b.id], junction_type="door_passage",
    )
    state.junctions.append(junction)
    for other in state.junctions:
        if wall.id in other.walls:
            other.walls.remove(wall.id)
            for c in (child_a, child_b):
                cl = c.centerline
                if min(cl.start.distance_to(other.point),
                       cl.end.distance_to(other.point)) < 1e-3:
                    other.walls.append(c.id)


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for door in state.doors:
        p = door.position
        cv2.circle(overlay, (int(p.x), int(p.y)), 4, (0, 200, 0), -1)
        if len(door.swing_arc) >= 2:
            points = np.asarray(
                [[round(point.x), round(point.y)] for point in door.swing_arc],
                dtype=np.int32,
            )
            cv2.polylines(overlay, [points], False, (0, 200, 0), 2)
        cv2.line(overlay, (int(p.x), int(p.y)),
                 (int(door.swing_end.x), int(door.swing_end.y)), (0, 200, 0), 1)
    return overlay
