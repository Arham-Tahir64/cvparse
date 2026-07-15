"""Deterministic geometry operations shared by import and human edits."""
from __future__ import annotations

import math

from .models import Coordinate


def distance(first: Coordinate, second: Coordinate) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def wall_polygon(
    start: Coordinate, end: Coordinate, thickness: float,
) -> list[Coordinate]:
    dx, dy = end.x - start.x, end.y - start.y
    length = max(1e-9, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    half = max(0.0, thickness) / 2.0
    return [
        Coordinate(start.x + nx * half, start.y + ny * half),
        Coordinate(end.x + nx * half, end.y + ny * half),
        Coordinate(end.x - nx * half, end.y - ny * half),
        Coordinate(start.x - nx * half, start.y - ny * half),
    ]


def wall_orientation(start: Coordinate, end: Coordinate) -> str:
    return "H" if abs(end.x - start.x) >= abs(end.y - start.y) else "V"


def point_at_offset(
    start: Coordinate, end: Coordinate, offset_px: float,
) -> Coordinate:
    length = distance(start, end)
    if length <= 1e-9:
        return start
    ux, uy = (end.x - start.x) / length, (end.y - start.y) / length
    return Coordinate(start.x + offset_px * ux, start.y + offset_px * uy)


def transform_wall_local_point(
    point: Coordinate,
    old_start: Coordinate,
    old_end: Coordinate,
    new_start: Coordinate,
    new_end: Coordinate,
) -> Coordinate:
    """Preserve a point's longitudinal/lateral coordinates on a changed wall."""
    old_length = max(distance(old_start, old_end), 1e-9)
    new_length = max(distance(new_start, new_end), 1e-9)
    old_ux = (old_end.x - old_start.x) / old_length
    old_uy = (old_end.y - old_start.y) / old_length
    new_ux = (new_end.x - new_start.x) / new_length
    new_uy = (new_end.y - new_start.y) / new_length
    rx, ry = point.x - old_start.x, point.y - old_start.y
    longitudinal = rx * old_ux + ry * old_uy
    lateral = rx * -old_uy + ry * old_ux
    return Coordinate(
        new_start.x + longitudinal * new_ux + lateral * -new_uy,
        new_start.y + longitudinal * new_uy + lateral * new_ux,
    )


def polygon_area(polygon: list[Coordinate]) -> float:
    if len(polygon) < 3:
        return 0.0
    points = polygon + [polygon[0]]
    twice_area = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:])
    )
    return abs(twice_area) / 2.0


def polygon_perimeter(polygon: list[Coordinate]) -> float:
    if len(polygon) < 2:
        return 0.0
    points = polygon + [polygon[0]]
    return sum(distance(first, second) for first, second in zip(points, points[1:]))
