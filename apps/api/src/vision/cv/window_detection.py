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
    binary = state.binary_masked if state.binary_masked is not None else state.binary

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
    state.debug.segment_counts["08_windows"] = len(windows)
    logger.info("detected %d windows", len(windows))

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "08_windows.png"),
            visualize(state, state.image),
        )
    return state


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

def _inner_line_windows(wall, candidates, tree, config, window_id_gen, gap_id_gen, state):
    cl = wall.centerline
    query_radius = cl.length / 2.0 + config.window_gap_max_px
    idx = tree.query_ball_point([cl.midpoint.x, cl.midpoint.y], query_radius)
    angle_tol = math.radians(config.window_inner_line_angle_tol_deg)
    max_perp = config.window_inner_line_perp_frac * wall.thickness

    intervals: list[tuple[float, float]] = []
    for i in idx:
        seg = candidates[i]
        if angle_diff_rad(seg.angle_rad, cl.angle_rad) > angle_tol:
            continue
        if point_to_line_distance(seg.midpoint, cl.start, cl.end) > max_perp:
            continue
        t0 = project_param(seg.start, cl.start, cl.end)
        t1 = project_param(seg.end, cl.start, cl.end)
        t0, t1 = sorted((t0, t1))
        if t0 < 0.0 or t1 > 1.0:
            continue
        intervals.append((t0, t1))

    merged = _merge_intervals(intervals, config.window_merge_overlap_ratio)
    results = []
    for t0, t1 in merged:
        center = point_at_param(cl.start, cl.end, (t0 + t1) / 2.0)
        width = (t1 - t0) * cl.length
        window = Window(id=window_id_gen(), position=center, width=width, wall_id=wall.id)
        results.append(window)
        state.gaps.append(_gap_record(gap_id_gen(), wall, center, width, 1.0))
    return results


def _merge_intervals(intervals, overlap_ratio_threshold):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for t0, t1 in intervals[1:]:
        last = merged[-1]
        overlap = min(last[1], t1) - max(last[0], t0)
        shorter = min(last[1] - last[0], t1 - t0)
        if shorter > 0 and overlap / shorter > overlap_ratio_threshold:
            last[1] = max(last[1], t1)
            last[0] = min(last[0], t0)
        else:
            merged.append([t0, t1])
    return [(a, b) for a, b in merged]


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
                window = Window(
                    id=window_id_gen(), position=center, width=run_px, wall_id=wall.id
                )
                results.append(window)
                state.gaps.append(_gap_record(gap_id_gen(), wall, center, run_px, fill))
        i = j + 1
    return results


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
