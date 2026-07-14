"""Module 08 - Window Detection.

Strategy A: inner-line matching against wall centerlines.
Strategy B: wall-face gap scanning for walls A did not cover, excluding
positions already recorded as door gaps by module 07.
Windows never split walls.
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import angle_diff_rad, point_at_param, point_to_line_distance, project_param
from .models import Gap, IdGenerator, PipelineState, Point, Wall, Window

logger = logging.getLogger("flowbuildr.cv.window_detection")

MODULE = "08_window_detection"


def run(state: PipelineState) -> PipelineState:
    config = state.config
    binary = (state.binary_cleaned if state.binary_cleaned is not None
              else state.binary_masked if state.binary_masked is not None
              else state.binary)

    window_id_gen = IdGenerator("WD")
    gap_id_gen = IdGenerator("G", start=len(state.gaps) + 1)

    windows: list[Window] = []
    covered_walls: set[str] = set()

    # Strategy A - inner-line matching
    inner_candidates = [
        s for s in state.classified_lines
        if s.classification == "unknown"
        and config.window_gap_min_px <= s.length <= config.window_gap_max_px
    ]
    if inner_candidates:
        tree = cKDTree([[s.midpoint.x, s.midpoint.y] for s in inner_candidates])
        for wall in state.walls:
            found = _inner_line_windows(
                wall, inner_candidates, tree, config, window_id_gen, gap_id_gen, state
            )
            if found:
                covered_walls.add(wall.id)
                windows.extend(found)

    # Strategy B - face-gap scanning for uncovered walls
    for wall in state.walls:
        if wall.id in covered_walls:
            continue
        windows.extend(
            _face_gap_windows(wall, binary, config, window_id_gen, gap_id_gen, state)
        )

    state.windows = windows
    _resolve_door_window_conflicts(state)
    _filter_nontangent_windows(state)
    _deduplicate_windows(state)
    state.debug.segment_counts["08_windows"] = len(state.windows)
    logger.info("detected %d windows", len(state.windows))

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "08_windows.png"),
            visualize(state, state.image),
        )
    return state


def _resolve_door_window_conflicts(state: PipelineState) -> None:
    """Prefer a supported window when one opening was also called a door.

    Door detection runs first so wall gaps cannot become windows. Strategy A,
    however, can independently find a framed exterior opening on a split wall
    with a different ID. If its span contains the alleged hinge and occupies a
    substantial fraction of the swing radius, exporting both classes creates
    a full false door sector. Preserve the split/opening, but remove the
    conflicting door object and its diagnostic door gap.
    """
    config = state.config
    conflicts = []
    for door in state.doors:
        for window in state.windows:
            distance = door.position.distance_to(window.position)
            if (
                distance <= config.door_window_conflict_radius_ratio * door.radius
                and distance <= config.door_window_conflict_width_ratio * window.width
            ):
                conflicts.append(door)
                break
    if not conflicts:
        state.debug.segment_counts["08_door_window_conflicts"] = 0
        return

    conflict_ids = {door.id for door in conflicts}
    state.suppressed_door_openings.extend(conflicts)
    state.doors = [door for door in state.doors if door.id not in conflict_ids]
    state.gaps = [
        gap for gap in state.gaps
        if not (
            gap.kind == "door"
            and any(gap.center.distance_to(door.position) <= 2.0 for door in conflicts)
        )
    ]
    state.debug.segment_counts["08_door_window_conflicts"] = len(conflicts)


def _filter_nontangent_windows(state: PipelineState) -> None:
    """Reject framed candidates whose supporting wall is not shell-tangent."""
    wall_lookup = {wall.id: wall for wall in state.walls}
    for wall in state.walls:
        for source_id in wall.source_ids:
            wall_lookup.setdefault(source_id, wall)
    rejected = [
        window for window in state.windows
        if (
            wall_lookup.get(window.wall_id) is None
            or not _has_exterior_tangent(
                wall_lookup[window.wall_id], window.position, state, state.config,
            )
        )
    ]
    rejected_ids = {window.id for window in rejected}
    state.windows = [
        window for window in state.windows if window.id not in rejected_ids
    ]
    state.gaps = [
        gap for gap in state.gaps
        if not (
            gap.kind == "window"
            and any(gap.center.distance_to(window.position) <= 2.0 for window in rejected)
        )
    ]
    state.debug.segment_counts["08_nontangent_windows"] = len(rejected)


def _deduplicate_windows(state: PipelineState) -> None:
    """Collapse one framed opening found on overlapping wall representations."""
    wall_lookup = {wall.id: wall for wall in state.walls}
    kept: list[Window] = []
    rejected: list[Window] = []
    for candidate in state.windows:
        candidate_wall = wall_lookup.get(candidate.wall_id)
        duplicate_index = None
        for index, existing in enumerate(kept):
            existing_wall = wall_lookup.get(existing.wall_id)
            if candidate_wall is None or existing_wall is None:
                continue
            width_ratio = min(candidate.width, existing.width) / max(
                1e-6, max(candidate.width, existing.width)
            )
            center_limit = max(
                state.config.window_inner_line_endpoint_tol_px,
                state.config.window_dedup_center_ratio
                * min(candidate.width, existing.width),
            )
            if (
                angle_diff_rad(
                    candidate_wall.centerline.angle_rad,
                    existing_wall.centerline.angle_rad,
                ) <= math.radians(state.config.window_inner_line_angle_tol_deg)
                and candidate.position.distance_to(existing.position) <= center_limit
                and width_ratio >= state.config.window_dedup_width_ratio
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_index]
        existing_wall = wall_lookup[existing.wall_id]
        # Prefer the more reliable/thicker structural representation; the
        # opening geometry itself remains source-derived in either case.
        candidate_score = (
            candidate_wall.merge_confidence,
            candidate_wall.fit_support_ratio,
            candidate_wall.thickness,
        )
        existing_score = (
            existing_wall.merge_confidence,
            existing_wall.fit_support_ratio,
            existing_wall.thickness,
        )
        if candidate_score > existing_score:
            kept[duplicate_index] = candidate
            rejected.append(existing)
        else:
            rejected.append(candidate)

    rejected_keys = {(window.wall_id, window.position.x, window.position.y)
                     for window in rejected}
    state.windows = kept
    state.gaps = [
        gap for gap in state.gaps
        if not (
            gap.kind == "window"
            and any(
                gap.wall_id == wall_id
                and gap.center.distance_to(Point(x, y)) <= 2.0
                for wall_id, x, y in rejected_keys
            )
        )
    ]
    state.debug.segment_counts["08_duplicate_windows"] = len(rejected)


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

def _inner_line_windows(wall, candidates, tree, config, window_id_gen, gap_id_gen, state):
    cl = wall.centerline
    query_radius = cl.length / 2.0 + config.window_gap_max_px
    idx = tree.query_ball_point([cl.midpoint.x, cl.midpoint.y], query_radius)
    angle_tol = math.radians(config.window_inner_line_angle_tol_deg)
    max_perp = config.window_inner_line_perp_frac * wall.thickness

    ux = (cl.end.x - cl.start.x) / max(1e-6, cl.length)
    uy = (cl.end.y - cl.start.y) / max(1e-6, cl.length)
    nx, ny = -uy, ux
    exact_intervals: list[tuple[float, float, float]] = []
    tolerant_intervals: list[tuple[float, float, float]] = []
    source_tolerant_intervals: list[tuple[float, float, float]] = []
    for i in idx:
        seg = candidates[i]
        if angle_diff_rad(seg.angle_rad, cl.angle_rad) > angle_tol:
            continue
        if point_to_line_distance(seg.midpoint, cl.start, cl.end) > max_perp:
            continue
        t0 = project_param(seg.start, cl.start, cl.end)
        t1 = project_param(seg.end, cl.start, cl.end)
        t0, t1 = sorted((t0, t1))
        endpoint_tolerance = (
            config.window_inner_line_endpoint_tol_px / max(1.0, cl.length)
        )
        if t0 < -endpoint_tolerance or t1 > 1.0 + endpoint_tolerance:
            continue
        exact = t0 >= 0.0 and t1 <= 1.0
        t0, t1 = max(0.0, t0), min(1.0, t1)
        offset = ((seg.midpoint.x - cl.midpoint.x) * nx +
                  (seg.midpoint.y - cl.midpoint.y) * ny)
        interval = (t0, t1, offset)
        if exact:
            exact_intervals.append(interval)
        else:
            tolerant_intervals.append(interval)
            if seg.id in wall.source_ids:
                source_tolerant_intervals.append(interval)

    # Do not let tolerance enlarge or duplicate an existing strict match. It
    # is only a fallback for a paired frame whose anti-aliased faces both
    # overrun a synthesized wall endpoint by a few pixels. Corroborating faces
    # must overlap, occupy distinct offsets, and have similar span lengths.
    merged = _frame_clusters(
        exact_intervals, config.window_merge_overlap_ratio,
        config.window_min_parallel_lines,
    )
    if not merged:
        source_corroborated = _corroborated_tolerant_intervals(
            source_tolerant_intervals, config,
        )
        merged = _frame_clusters(
            source_corroborated, config.window_merge_overlap_ratio,
            max(2, config.window_min_parallel_lines),
        )
    if not merged:
        # Paired-wall synthesis can choose alternate long faces at a crowded
        # exterior opening. Three or more agreeing offsets are stronger frame
        # evidence than the two source-ID fallback and remain independent of
        # any one wall representation.
        repeated = _corroborated_tolerant_intervals(tolerant_intervals, config)
        merged = _frame_clusters(
            repeated, config.window_merge_overlap_ratio,
            max(config.window_repeated_frame_min_lines,
                config.window_min_parallel_lines),
        )
    results = []
    for t0, t1 in merged:
        center = point_at_param(cl.start, cl.end, (t0 + t1) / 2.0)
        if not _has_exterior_context(wall, center, state, config):
            continue
        width = (t1 - t0) * cl.length
        window = Window(id=window_id_gen(), position=center, width=width, wall_id=wall.id)
        results.append(window)
        state.gaps.append(_gap_record(gap_id_gen(), wall, center, width, 1.0))
    return results


def _corroborated_tolerant_intervals(intervals, config):
    corroborated = []
    for index, candidate in enumerate(intervals):
        t0, t1, offset = candidate
        length = t1 - t0
        for other_index, other in enumerate(intervals):
            if index == other_index or abs(offset - other[2]) < 1.0:
                continue
            other_length = other[1] - other[0]
            overlap = min(t1, other[1]) - max(t0, other[0])
            length_ratio = min(length, other_length) / max(
                1e-6, max(length, other_length)
            )
            if (
                overlap > config.window_merge_overlap_ratio
                * min(length, other_length)
                and length_ratio >= config.window_tolerant_frame_length_ratio
            ):
                corroborated.append(candidate)
                break
    return corroborated


def _frame_clusters(intervals, overlap_ratio_threshold, min_parallel_lines):
    if not intervals:
        return []
    clusters: list[list[tuple[float, float, float]]] = []
    for candidate in sorted(intervals):
        t0, t1, _ = candidate
        match = None
        for cluster in clusters:
            c0 = min(item[0] for item in cluster)
            c1 = max(item[1] for item in cluster)
            overlap = min(c1, t1) - max(c0, t0)
            shorter = min(c1 - c0, t1 - t0)
            if shorter > 0 and overlap / shorter > overlap_ratio_threshold:
                match = cluster
                break
        if match is None:
            clusters.append([candidate])
        else:
            match.append(candidate)

    output = []
    for cluster in clusters:
        distinct_offsets = []
        for _, _, offset in cluster:
            if all(abs(offset - existing) >= 1.0 for existing in distinct_offsets):
                distinct_offsets.append(offset)
        if len(distinct_offsets) < min_parallel_lines:
            continue
        output.append((
            min(item[0] for item in cluster),
            max(item[1] for item in cluster),
        ))
    return output


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

def _face_gap_windows(wall, binary, config, window_id_gen, gap_id_gen, state):
    cl = wall.centerline
    length = cl.length
    if length < 1:
        return []
    n = max(config.window_scan_min_samples, int(length))
    ux = (cl.end.x - cl.start.x) / length
    uy = (cl.end.y - cl.start.y) / length
    nx, ny = -uy, ux
    half_t = wall.thickness / 2.0
    band = config.window_gap_scan_half_band_px
    h, w = binary.shape[:2]

    door_ranges = _door_param_ranges(wall, cl, state)

    def face_has_ink(px, py) -> bool:
        for d in range(-band, band + 1):
            x, y = int(round(px + d * nx)), int(round(py + d * ny))
            if 0 <= x < w and 0 <= y < h and binary[y, x] > 0:
                return True
        return False

    ts = np.linspace(0.0, 1.0, n)
    open_flags = []
    for t in ts:
        if any(lo <= t <= hi for lo, hi in door_ranges):
            open_flags.append(False)  # door territory: never a window sample
            continue
        cx = cl.start.x + t * (cl.end.x - cl.start.x)
        cy = cl.start.y + t * (cl.end.y - cl.start.y)
        face_a = face_has_ink(cx + half_t * nx, cy + half_t * ny)
        face_b = face_has_ink(cx - half_t * nx, cy - half_t * ny)
        open_flags.append(not face_a and not face_b)

    results = []
    step_px = length / (n - 1)
    i = 0
    while i < n:
        if not open_flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and open_flags[j + 1]:
            j += 1
        run_px = (j - i + 1) * step_px
        if config.window_gap_min_px <= run_px <= config.window_gap_max_px:
            t_mid = (ts[i] + ts[j]) / 2.0
            fill = _side_fill(cl, ts[i], ts[j], wall, binary, config, (ux, uy), (nx, ny))
            if fill is not None:
                center = point_at_param(cl.start, cl.end, t_mid)
                if not _has_exterior_context(wall, center, state, config):
                    i = j + 1
                    continue
                window = Window(
                    id=window_id_gen(), position=center, width=run_px, wall_id=wall.id
                )
                results.append(window)
                state.gaps.append(_gap_record(gap_id_gen(), wall, center, run_px, fill))
        i = j + 1
    return results


def _has_exterior_context(wall, center, state, config) -> bool:
    context_rooms = getattr(state, "window_context_rooms", None) or state.rooms
    if not config.window_require_exterior_context or not context_rooms:
        return True
    cl = wall.centerline
    length = max(1e-6, cl.length)
    ux = (cl.end.x - cl.start.x) / length
    uy = (cl.end.y - cl.start.y) / length
    nx, ny = -uy, ux
    offset = wall.thickness / 2.0 + config.window_exterior_sample_px
    samples = [
        (center.x + nx * offset, center.y + ny * offset),
        (center.x - nx * offset, center.y - ny * offset),
    ]

    def in_any_room(point):
        for room in context_rooms:
            if len(room.polygon) < 3:
                continue
            contour = np.asarray(
                [[p.x, p.y] for p in room.polygon], dtype=np.float32
            )
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return True
        return False

    inside = [in_any_room(point) for point in samples]
    if inside[0] == inside[1]:
        return False

    hull_points = np.asarray(
        [[point.x, point.y] for room in context_rooms for point in room.polygon],
        dtype=np.float32,
    )
    if len(hull_points) < 3:
        return False
    hull = cv2.convexHull(hull_points.reshape(-1, 1, 2))
    distance = abs(cv2.pointPolygonTest(hull, (center.x, center.y), True))
    return distance <= config.window_exterior_hull_dist_px


def _has_exterior_tangent(wall, center, state, config) -> bool:
    context_rooms = getattr(state, "window_context_rooms", None) or state.rooms
    if not config.window_require_exterior_context or not context_rooms:
        return True
    hull_points = np.asarray(
        [[point.x, point.y] for room in context_rooms for point in room.polygon],
        dtype=np.float32,
    )
    if len(hull_points) < 3:
        return False
    hull = cv2.convexHull(hull_points.reshape(-1, 1, 2))

    # Distance alone admits short interior/fixture lines near the shell. A
    # window span must also be tangent to the nearest exterior hull edge.
    vertices = [Point(float(item[0][0]), float(item[0][1])) for item in hull]
    edge = min(
        zip(vertices, vertices[1:] + vertices[:1]),
        key=lambda pair: point_to_line_distance(center, pair[0], pair[1]),
    )
    edge_angle = math.atan2(edge[1].y - edge[0].y, edge[1].x - edge[0].x)
    return angle_diff_rad(wall.centerline.angle_rad, edge_angle) <= math.radians(
        config.window_exterior_tangent_angle_tol_deg
    )


def _door_param_ranges(wall, cl, state):
    ranges = []
    for gap in state.gaps:
        if gap.kind != "door" or gap.wall_id != wall.id:
            continue
        t = project_param(gap.center, cl.start, cl.end)
        half = (gap.width_px / cl.length) if cl.length > 0 else 0.0
        ranges.append((t - half, t + half))
    return ranges


def _side_fill(cl, t0, t1, wall, binary, config, direction, normal):
    """Verify wall ink exists beside the gap; None if both sides empty.

    Returns the measured side fill ratio for the gap record.
    """
    ux, uy = direction
    nx, ny = -uy, ux
    half_t = wall.thickness / 2.0
    band = config.window_gap_scan_half_band_px
    sample_px = config.window_gap_side_sample_px
    h, w = binary.shape[:2]
    length = cl.length

    def fill_at(t_edge, sign) -> float:
        hits = total = 0
        for k in range(1, sample_px + 1):
            t = t_edge + sign * (k / length)
            if not (0.0 <= t <= 1.0):
                break
            cx = cl.start.x + t * (cl.end.x - cl.start.x)
            cy = cl.start.y + t * (cl.end.y - cl.start.y)
            for face_sign in (1, -1):
                total += 1
                px, py = cx + face_sign * half_t * nx, cy + face_sign * half_t * ny
                for d in range(-band, band + 1):
                    x, y = int(round(px + d * nx)), int(round(py + d * ny))
                    if 0 <= x < w and 0 <= y < h and binary[y, x] > 0:
                        hits += 1
                        break
        return hits / total if total else 0.0

    left = fill_at(t0, -1)
    right = fill_at(t1, +1)
    # a genuine window opening has wall ink on both sides; a wall end or
    # corner has an empty side (spec test criterion 9)
    if left < config.window_gap_min_side_fill or right < config.window_gap_min_side_fill:
        return None
    return min(left, right)


def _gap_record(gid, wall, center: Point, width: float, score: float) -> Gap:
    half = width / 2.0
    return Gap(
        id=gid, wall_id=wall.id,
        orientation="V" if wall.orientation == "V" else "H",
        center=center, width_px=width,
        bbox=(center.x - half, center.y - half, center.x + half, center.y + half),
        kind="window", wall_break_score=score, opening_fill_ratio=1.0,
    )


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for win in state.windows:
        p = win.position
        cv2.circle(overlay, (int(p.x), int(p.y)), 4, (0, 165, 255), -1)
    return overlay
