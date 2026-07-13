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

from .geometry import point_at_param, point_to_line_distance, project_param
from .junction_snapping import split_wall_at
from .models import Door, Gap, IdGenerator, Junction, PipelineState, Point, Wall

logger = logging.getLogger("flowbuildr.cv.door_detection")

MODULE = "07_door_detection"

_ARC_SAMPLES = 72  # points sampled around each candidate circle


def run(state: PipelineState) -> PipelineState:
    config = state.config
    binary = state.binary_masked if state.binary_masked is not None else state.binary

    erased = _erase_walls(binary, state.walls, config)
    circles = _hough_circles(erased, config)
    logger.debug("hough circle candidates: %d", len(circles))

    door_id_gen = IdGenerator("D")
    gap_id_gen = IdGenerator("G", start=len(state.gaps) + 1)
    wall_id_gen = IdGenerator("W", start=_max_wall_id(state.walls) + 1)
    junction_id_gen = IdGenerator("J", start=len(state.junctions) + 1)

    candidates = []
    for cx, cy, radius in circles:
        coverage, mid_angle = _arc_coverage(erased, cx, cy, radius)
        if not (config.arc_coverage_min <= coverage <= config.arc_coverage_max):
            continue
        wall = _nearest_wall(Point(cx, cy), state.walls, config)
        if wall is None:
            continue
        candidates.append((radius, coverage, mid_angle, cx, cy, wall))

    # dedup: larger radius wins, hinges within door_dedup_dist_px collapse
    candidates.sort(key=lambda c: c[0], reverse=True)
    accepted: list[tuple] = []
    doors: list[Door] = []
    for radius, coverage, mid_angle, cx, cy, wall in candidates:
        cl = wall.centerline
        t = min(1.0, max(0.0, project_param(Point(cx, cy), cl.start, cl.end)))
        hinge = point_at_param(cl.start, cl.end, t)
        if any(hinge.distance_to(d.position) <= config.door_dedup_dist_px for d in doors):
            continue

        swing_end = Point(cx + radius * math.cos(mid_angle),
                          cy + radius * math.sin(mid_angle))
        swing = _swing_direction(cl, hinge, swing_end)
        door = Door(
            id=door_id_gen(), position=hinge, swing_end=swing_end, radius=float(radius),
            wall_id=wall.id, swing_direction=swing,
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

    Returns (coverage_fraction, midpoint_angle_of_run).
    """
    h, w = image.shape[:2]
    angles = np.linspace(0, 2 * math.pi, _ARC_SAMPLES, endpoint=False)
    on = []
    for a in angles:
        hit = False
        # tolerate 1px radial jitter
        for dr in (-1, 0, 1):
            x = int(round(cx + (radius + dr) * math.cos(a)))
            y = int(round(cy + (radius + dr) * math.sin(a)))
            if 0 <= x < w and 0 <= y < h and image[y, x] > 0:
                hit = True
                break
        on.append(hit)

    if not any(on):
        return 0.0, 0.0
    if all(on):
        return 1.0, 0.0

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
    mid_idx = (best_start + best_len // 2) % _ARC_SAMPLES
    return best_len / _ARC_SAMPLES, float(angles[mid_idx])


def _nearest_wall(center: Point, walls, config):
    best, best_dist = None, float("inf")
    for wall in walls:
        cl = wall.centerline
        t = project_param(center, cl.start, cl.end)
        t = min(1.0, max(0.0, t))
        dist = center.distance_to(point_at_param(cl.start, cl.end, t))
        if dist < best_dist:
            best, best_dist = wall, dist
    if best is None or best_dist > config.door_wall_snap_px + best.thickness / 2.0:
        return None
    return best


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
        cv2.circle(overlay, (int(p.x), int(p.y)), int(door.radius), (0, 200, 0), 1)
        cv2.line(overlay, (int(p.x), int(p.y)),
                 (int(door.swing_end.x), int(door.swing_end.y)), (0, 200, 0), 1)
    return overlay
