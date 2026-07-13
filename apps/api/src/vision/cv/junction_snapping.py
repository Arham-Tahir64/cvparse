"""Module 06 - Junction Snapping.

Cluster near-coincident wall endpoints into junctions, classify junction
topology, close small gaps, split walls where needed, and enforce the
invariant that every wall endpoint coincides with a junction.
"""
from __future__ import annotations

import dataclasses
import logging
import os

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .geometry import point_at_param, project_param, segment_orientation
from .models import (
    IdGenerator, Junction, LineSegment, PipelineError, PipelineState, Point, Wall,
)

logger = logging.getLogger("flowbuildr.cv.junction_snapping")

MODULE = "06_junction_snapping"

_TYPE_BY_COUNT = {1: "dead_end", 2: "L", 3: "T", 4: "X"}


def _junction_type(wall_count: int) -> str:
    return _TYPE_BY_COUNT.get(wall_count, "Y")


def run(state: PipelineState) -> PipelineState:
    config = state.config
    if not state.walls:
        state.junctions = []
        return state

    wall_id_gen = IdGenerator("W", start=_max_id(state.walls) + 1)
    junction_id_gen = IdGenerator("J")

    _cluster_endpoints(state, junction_id_gen)
    _close_gaps(state, wall_id_gen, junction_id_gen)
    _remove_zero_length_walls(state)
    _check_invariant(state)

    state.debug.segment_counts["06_junctions"] = len(state.junctions)

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "06_junctions.png"),
            visualize(state, state.image),
        )
    return state


def _max_id(walls: list[Wall]) -> int:
    best = 0
    for w in walls:
        try:
            best = max(best, int(w.id[1:]))
        except ValueError:
            pass
    return best


def _cluster_endpoints(state: PipelineState, junction_id_gen) -> None:
    config = state.config
    points = []
    owners = []  # (wall_index, which_end)
    for wi, wall in enumerate(state.walls):
        cl = wall.centerline
        points.append([cl.start.x, cl.start.y])
        owners.append((wi, "start"))
        points.append([cl.end.x, cl.end.y])
        owners.append((wi, "end"))

    labels = DBSCAN(eps=config.junction_snap_radius_px, min_samples=1).fit_predict(
        np.array(points)
    )

    junctions: list[Junction] = []
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        centroid = np.array(points)[idx].mean(axis=0)
        point = Point(float(centroid[0]), float(centroid[1]))
        wall_ids = []
        for k in idx:
            wi, end = owners[k]
            wall = state.walls[wi]
            if wall.id not in wall_ids:
                wall_ids.append(wall.id)
            _set_wall_end(state.walls, wi, end, point)
        junctions.append(Junction(
            id=junction_id_gen(), point=point, walls=wall_ids,
            junction_type=_junction_type(len(wall_ids)),
        ))
    state.junctions = junctions


def _set_wall_end(walls: list[Wall], wi: int, end: str, point: Point) -> None:
    wall = walls[wi]
    cl = wall.centerline
    new_cl = dataclasses.replace(cl, **{end: point})
    wall.centerline = new_cl
    wall.length_px = new_cl.length
    wall.orientation = segment_orientation(new_cl)


def _close_gaps(state: PipelineState, wall_id_gen, junction_id_gen) -> None:
    config = state.config
    for iteration in range(config.gap_closure_max_iterations):
        if not _close_gaps_once(state, wall_id_gen, junction_id_gen):
            return
    logger.warning("gap closure hit iteration cap (%d)", config.gap_closure_max_iterations)


def _close_gaps_once(state: PipelineState, wall_id_gen, junction_id_gen) -> bool:
    config = state.config
    for junction in state.junctions:
        if len(junction.walls) > 1:
            continue
        # isolated endpoint: try projecting onto another wall's centerline body
        best = None
        for target in state.walls:
            if target.id in junction.walls:
                continue
            cl = target.centerline
            t = project_param(junction.point, cl.start, cl.end)
            if not (0.0 < t < 1.0):
                continue
            proj = point_at_param(cl.start, cl.end, t)
            dist = junction.point.distance_to(proj)
            if dist < config.gap_closure_max_px and (best is None or dist < best[0]):
                best = (dist, target, t, proj)
        if best is None:
            continue

        _, target, t, proj = best
        _snap_junction_to(state, junction, proj)
        _split_wall(state, target, t, junction, wall_id_gen)
        junction.junction_type = "T"
        return True
    return False


def _snap_junction_to(state: PipelineState, junction: Junction, point: Point) -> None:
    old = junction.point
    junction.point = point
    for wi, wall in enumerate(state.walls):
        if wall.id not in junction.walls:
            continue
        cl = wall.centerline
        if cl.start.distance_to(old) < 1e-6:
            _set_wall_end(state.walls, wi, "start", point)
        if cl.end.distance_to(old) < 1e-6:
            _set_wall_end(state.walls, wi, "end", point)


def split_wall_at(
    wall: Wall, split_point: Point, wall_id_gen
) -> tuple[Wall, Wall]:
    """Split `wall` at `split_point`, propagating all provenance fields.

    Shared with module 07 (door splits).
    """
    def child(start: Point, end: Point) -> Wall:
        cl = LineSegment(start, end, thickness=wall.centerline.thickness)
        return Wall(
            id=wall_id_gen(),
            orientation=segment_orientation(cl),
            centerline=cl,
            thickness=wall.thickness,
            visual_thickness=wall.visual_thickness,
            wall_type=wall.wall_type,
            merge_kind=wall.merge_kind,
            fit_support_ratio=wall.fit_support_ratio,
            merge_confidence=wall.merge_confidence,
            source_ids=list(dict.fromkeys([*wall.source_ids, wall.id])),
            length_px=cl.length,
        )

    return (
        child(wall.centerline.start, split_point),
        child(split_point, wall.centerline.end),
    )


def _split_wall(
    state: PipelineState, target: Wall, t: float, junction: Junction, wall_id_gen
) -> None:
    child_a, child_b = split_wall_at(target, junction.point, wall_id_gen)
    idx = state.walls.index(target)
    state.walls[idx: idx + 1] = [child_a, child_b]

    # rewire junction membership: replace target with the touching child
    junction.walls = [w for w in junction.walls] + [child_a.id, child_b.id]
    for other in state.junctions:
        if other is junction or target.id not in other.walls:
            continue
        other.walls.remove(target.id)
        for c in (child_a, child_b):
            cl = c.centerline
            if min(cl.start.distance_to(other.point), cl.end.distance_to(other.point)) < 1e-3:
                other.walls.append(c.id)
    junction.walls = [w for w in junction.walls if w != target.id]


def _remove_zero_length_walls(state: PipelineState) -> None:
    config = state.config
    keep = [w for w in state.walls if w.centerline.length >= config.zero_length_wall_px]
    removed = {w.id for w in state.walls} - {w.id for w in keep}
    if removed:
        logger.debug("removed %d zero-length walls", len(removed))
        state.walls = keep
        for j in state.junctions:
            j.walls = [wid for wid in j.walls if wid not in removed]
        state.junctions = [j for j in state.junctions if j.walls]
        for j in state.junctions:
            if j.junction_type != "door_passage":
                j.junction_type = _junction_type(len(j.walls))


def _check_invariant(state: PipelineState) -> None:
    tol = state.config.junction_coincidence_tol_px
    points = [j.point for j in state.junctions]
    for wall in state.walls:
        for endpoint in (wall.centerline.start, wall.centerline.end):
            if not any(endpoint.distance_to(p) <= tol for p in points):
                raise PipelineError(
                    MODULE,
                    f"wall {wall.id} endpoint ({endpoint.x:.1f}, {endpoint.y:.1f}) "
                    "does not coincide with any junction",
                )


_RADIUS_BY_TYPE = {"dead_end": 3, "L": 5, "T": 7, "X": 9, "Y": 9, "door_passage": 5}


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for wall in state.walls:
        cl = wall.centerline
        cv2.line(overlay, (int(cl.start.x), int(cl.start.y)),
                 (int(cl.end.x), int(cl.end.y)), (0, 200, 0), 2)
    for j in state.junctions:
        cv2.circle(overlay, (int(j.point.x), int(j.point.y)),
                   _RADIUS_BY_TYPE.get(j.junction_type, 5), (255, 0, 0), 2)
    return overlay
