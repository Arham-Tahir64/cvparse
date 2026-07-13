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
    wall_mask = np.zeros(shape, np.uint8)
    door_mask = np.zeros(shape, np.uint8)
    window_mask = np.zeros(shape, np.uint8)
    room_mask = np.zeros(shape, np.uint8)

    for wall in state.walls:
        _draw_wall_band(wall_mask, wall, 255)

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
                wall_mask, _pixel(door.position), _pixel(door.swing_arc[0]), 0,
                _wall_thickness(wall),
            )

    # Windows own their footprint even though they lie within a wall.
    wall_mask[window_mask > 0] = 0

    for room in state.rooms:
        if len(room.polygon) < 3:
            continue
        polygon = np.asarray([[_pixel(point) for point in room.polygon]], np.int32)
        cv2.fillPoly(room_mask, polygon, 255)

    combined = np.zeros((*shape, 3), np.uint8)
    combined[room_mask > 0] = CLASS_COLORS["room"]
    combined[wall_mask > 0] = CLASS_COLORS["wall"]
    combined[door_mask > 0] = CLASS_COLORS["door"]
    combined[window_mask > 0] = CLASS_COLORS["window"]

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
