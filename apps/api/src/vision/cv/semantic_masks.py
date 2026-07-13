"""Build explicit class masks after structural inference.

Walls are filled at their measured face-to-face thickness. Window spans are
aligned with and cut out of their supporting walls, so the two classes remain
separate in both binary exports and the combined color mask.
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np

from .geometry import point_at_param, project_param
from .models import PipelineState, Point, Wall

logger = logging.getLogger("flowbuildr.cv.semantic_masks")

MODULE = "13_semantic_masks"

# BGR debug colors matching the PDF/SVG semantic palette.
CLASS_COLORS = {
    "room": (232, 199, 173),
    "wall": (41, 38, 214),
    "door": (43, 161, 43),
    "window": (180, 119, 31),
}


def run(state: PipelineState) -> PipelineState:
    if state.image is None:
        return state
    shape = state.image.shape[:2]
    wall_boundaries = np.zeros(shape, np.uint8)
    wall_polygons = np.zeros(shape, np.uint8)
    door_mask = np.zeros(shape, np.uint8)
    window_mask = np.zeros(shape, np.uint8)
    room_mask = np.zeros(shape, np.uint8)

    for room in state.rooms:
        if len(room.polygon) < 3:
            continue
        polygon = np.asarray([[_pixel(point) for point in room.polygon]], np.int32)
        cv2.fillPoly(room_mask, polygon, 255)

    line_lookup = {line.id: line for line in state.classified_lines}
    for wall in state.walls:
        if state.config.manhattan and wall.orientation == "diagonal":
            continue
        _draw_wall_band(wall_polygons, wall, 255)
        sources = [line_lookup[source_id] for source_id in wall.source_ids
                   if source_id in line_lookup]
        if sources:
            for source in sources:
                cv2.line(
                    wall_boundaries, _pixel(source.start), _pixel(source.end),
                    255, max(1, int(round(source.thickness))),
                )
        else:
            # Synthetic/programmatic walls may not retain source edges.
            contour = np.zeros(shape, np.uint8)
            _draw_wall_band(contour, wall, 255)
            wall_boundaries = cv2.bitwise_or(
                wall_boundaries,
                cv2.morphologyEx(contour, cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)),
            )

    # The proposal pass sees complete faces that the cleaned LSD pass can
    # fragment. Both masks are independently supported structural geometry;
    # their union recovers missing runs without admitting arbitrary image ink.
    reconstructed = wall_polygons.copy()
    if state.structural_protection_mask is not None:
        protection = state.structural_protection_mask
        if state.config.manhattan:
            run = max(3, int(state.config.wall_region_axis_min_run_px))
            horizontal = cv2.morphologyEx(
                protection, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)
            )
            vertical = cv2.morphologyEx(
                protection, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8)
            )
            protection = cv2.bitwise_or(horizontal, vertical)
        reconstructed = cv2.bitwise_or(
            reconstructed, protection,
        )
    if state.semantic_plan_mask is not None:
        reconstructed = cv2.bitwise_and(reconstructed, state.semantic_plan_mask)

    exterior_ring = _exterior_wall_ring(state, room_mask)
    if exterior_ring is not None:
        reconstructed = cv2.bitwise_or(reconstructed, exterior_ring)

    gap = max(1, int(state.config.wall_region_gap_close_px))
    horizontal = cv2.morphologyEx(
        reconstructed, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8)
    )
    vertical = cv2.morphologyEx(
        reconstructed, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8)
    )
    wall_mask = cv2.bitwise_or(reconstructed, cv2.bitwise_or(horizontal, vertical))

    for window in state.windows:
        wall = _wall_for_id(state.walls, window.wall_id)
        if wall is None:
            continue
        ux, uy = _unit(wall)
        half = window.width / 2.0
        start = Point(window.position.x - half * ux, window.position.y - half * uy)
        end = Point(window.position.x + half * ux, window.position.y + half * uy)
        thickness = _wall_thickness(wall)
        cv2.line(window_mask, _pixel(start), _pixel(end), 255, thickness)

    door_openings = np.zeros(shape, np.uint8)
    for door in state.doors:
        if len(door.swing_arc) >= 2:
            polygon = np.asarray(
                [[_pixel(door.position)] + [_pixel(point) for point in door.swing_arc]],
                dtype=np.int32,
            )
            cv2.fillPoly(door_mask, polygon, 255)
        wall = _wall_for_door(state.walls, door)
        if wall is not None and door.swing_arc:
            # First arc sample is the closed/jamb endpoint; clear exactly the
            # physical opening from the wall footprint.
            cv2.line(
                door_openings, _pixel(door.position), _pixel(door.swing_arc[0]),
                255, _wall_thickness(wall),
            )

    # Openings own their footprints even though they lie within reconstructed
    # wall corridors. Subtract after repair so closure cannot bridge them.
    wall_mask[door_openings > 0] = 0
    wall_mask[door_mask > 0] = 0
    wall_mask[window_mask > 0] = 0

    combined = np.zeros((*shape, 3), np.uint8)
    combined[room_mask > 0] = CLASS_COLORS["room"]
    combined[wall_mask > 0] = CLASS_COLORS["wall"]
    combined[door_mask > 0] = CLASS_COLORS["door"]
    combined[window_mask > 0] = CLASS_COLORS["window"]

    state.wall_boundary_mask = wall_boundaries
    state.wall_polygon_mask = wall_polygons
    state.wall_repaired_mask = wall_mask
    state.wall_mask = wall_mask
    state.door_mask = door_mask
    state.window_mask = window_mask
    state.room_region_mask = room_mask
    state.combined_class_mask = combined
    state.debug.segment_counts["13_wall_pixels"] = int(np.count_nonzero(wall_mask))
    state.debug.segment_counts["13_window_pixels"] = int(np.count_nonzero(window_mask))

    if state.config.debug_visualize and state.config.debug_output_dir:
        out_dir = os.path.join(state.config.debug_output_dir, MODULE)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "wall_boundaries.png"), wall_boundaries)
        cv2.imwrite(os.path.join(out_dir, "wall_polygons.png"), wall_polygons)
        if exterior_ring is not None:
            cv2.imwrite(os.path.join(out_dir, "exterior_wall_ring.png"), exterior_ring)
        cv2.imwrite(os.path.join(out_dir, "wall_repaired_mask.png"), wall_mask)
        cv2.imwrite(os.path.join(out_dir, "wall_mask.png"), wall_mask)
        cv2.imwrite(os.path.join(out_dir, "door_mask.png"), door_mask)
        cv2.imwrite(os.path.join(out_dir, "window_mask.png"), window_mask)
        cv2.imwrite(os.path.join(out_dir, "room_region_mask.png"), room_mask)
        cv2.imwrite(os.path.join(out_dir, "combined_class_mask.png"), combined)

    logger.info(
        "semantic masks: %d wall pixels, %d window pixels",
        np.count_nonzero(wall_mask), np.count_nonzero(window_mask),
    )
    return state


def _draw_wall_band(mask: np.ndarray, wall: Wall, value: int) -> None:
    cv2.line(
        mask, _pixel(wall.centerline.start), _pixel(wall.centerline.end), value,
        _wall_thickness(wall),
    )


def _exterior_wall_ring(
    state: PipelineState, room_mask: np.ndarray,
) -> np.ndarray | None:
    """Recover a rectangular exterior wall from observed inner/outer faces.

    Seeded room regions terminate at the inner face. In Manhattan plans, a
    sustained parallel ink run just outside each room-envelope side supplies
    the outer face. Every side must be independently supported; otherwise no
    rectangle is inferred (important for L-shaped or incomplete plans).
    """
    if not state.config.manhattan or not np.any(room_mask):
        return None
    if state.structural_core_mask is not None:
        core = state.structural_core_mask > 0
        core_x, core_y, core_width, core_height = cv2.boundingRect(
            core.astype(np.uint8)
        )
        if core_width < 1 or core_height < 1:
            return None
        rectangularity = float(np.mean(
            core[core_y:core_y + core_height, core_x:core_x + core_width]
        ))
        if rectangularity < state.config.exterior_wall_min_rectangularity:
            logger.info(
                "exterior rectangle skipped: structural rectangularity %.3f",
                rectangularity,
            )
            return None
    binary = (state.binary_cleaned if state.binary_cleaned is not None
              else state.binary_masked if state.binary_masked is not None
              else state.binary)
    if binary is None:
        return None

    x, y, width, height = cv2.boundingRect(room_mask)
    if width < 3 or height < 3:
        return None
    right = x + width - 1
    bottom = y + height - 1
    search = max(1, int(round(state.config.wall_thickness_max_px)))
    threshold = float(state.config.exterior_wall_min_side_support)

    left_outer = _outer_supported_coordinate(
        binary[y:y + height, :], range(max(0, x - search), x + 1),
        threshold, choose_min=True,
    )
    right_outer = _outer_supported_coordinate(
        binary[y:y + height, :],
        range(right, min(binary.shape[1] - 1, right + search) + 1),
        threshold, choose_min=False,
    )
    top_outer = _outer_supported_coordinate(
        binary[:, x:x + width].T, range(max(0, y - search), y + 1),
        threshold, choose_min=True,
    )
    bottom_outer = _outer_supported_coordinate(
        binary[:, x:x + width].T,
        range(bottom, min(binary.shape[0] - 1, bottom + search) + 1),
        threshold, choose_min=False,
    )
    if None in (left_outer, right_outer, top_outer, bottom_outer):
        return None

    offsets = (x - left_outer, right_outer - right,
               y - top_outer, bottom_outer - bottom)
    if any(offset < state.config.wall_thickness_min_px for offset in offsets):
        return None

    # Exterior shells are normally specified consistently even when a faint
    # or window-interrupted face is detected on only one side. The second
    # largest independently measured offset is robust to one spurious distant
    # line and supplies the missing opposite face on weaker sides.
    shell = int(sorted(offsets)[-2])
    shell = min(shell, search)
    left_outer = max(0, x - shell)
    right_outer = min(binary.shape[1] - 1, right + shell)
    top_outer = max(0, y - shell)
    bottom_outer = min(binary.shape[0] - 1, bottom + shell)

    ring = np.zeros_like(room_mask)
    cv2.rectangle(
        ring, (left_outer, top_outer), (right_outer, bottom_outer), 255,
        cv2.FILLED,
    )
    cv2.rectangle(
        ring, (x + 1, y + 1), (right - 1, bottom - 1), 0, cv2.FILLED,
    )
    logger.info("exterior wall offsets measured=%s; reconstructed shell=%d",
                offsets, shell)
    return ring


def _outer_supported_coordinate(
    side_image: np.ndarray, positions, threshold: float, choose_min: bool,
) -> int | None:
    span = max(1, side_image.shape[0])
    supported = [
        position for position in positions
        if np.count_nonzero(side_image[:, position]) / span >= threshold
    ]
    if not supported:
        return None
    return min(supported) if choose_min else max(supported)


def _wall_thickness(wall: Wall) -> int:
    return max(1, int(round(max(wall.thickness, wall.visual_thickness))))


def _unit(wall: Wall) -> tuple[float, float]:
    cl = wall.centerline
    length = max(1e-6, cl.length)
    return (cl.end.x - cl.start.x) / length, (cl.end.y - cl.start.y) / length


def _pixel(point: Point) -> tuple[int, int]:
    return int(round(point.x)), int(round(point.y))


def _wall_for_id(walls: list[Wall], wall_id: str) -> Wall | None:
    return next(
        (wall for wall in walls
         if wall.id == wall_id or wall_id in wall.source_ids),
        None,
    )


def _wall_for_door(walls, door) -> Wall | None:
    direct = _wall_for_id(walls, door.wall_id)
    if direct is not None:
        return direct
    best, best_distance = None, math.inf
    for wall in walls:
        cl = wall.centerline
        t = min(1.0, max(0.0, project_param(door.position, cl.start, cl.end)))
        point = point_at_param(cl.start, cl.end, t)
        distance = point.distance_to(door.position)
        if distance < best_distance:
            best, best_distance = wall, distance
    return best
