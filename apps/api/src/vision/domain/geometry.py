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


def project_point_to_segment(
    point: Coordinate,
    start: Coordinate,
    end: Coordinate,
) -> tuple[Coordinate, float, float]:
    """Return clamped projection, longitudinal offset, and lateral distance."""
    length = distance(start, end)
    if length <= 1e-9:
        return start, 0.0, distance(point, start)
    ux = (end.x - start.x) / length
    uy = (end.y - start.y) / length
    offset = min(
        length,
        max(0.0, (point.x - start.x) * ux + (point.y - start.y) * uy),
    )
    projected = Coordinate(start.x + offset * ux, start.y + offset * uy)
    return projected, offset, distance(point, projected)


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


def _cross(first: Coordinate, second: Coordinate, point: Coordinate) -> float:
    return (
        (second.x - first.x) * (point.y - first.y)
        - (second.y - first.y) * (point.x - first.x)
    )


def point_on_segment(
    point: Coordinate,
    start: Coordinate,
    end: Coordinate,
    tolerance: float = 1e-6,
) -> bool:
    """Return whether a point lies on a finite segment within pixel tolerance."""
    scale = max(1.0, distance(start, end))
    if abs(_cross(start, end, point)) > tolerance * scale:
        return False
    return (
        min(start.x, end.x) - tolerance <= point.x <= max(start.x, end.x) + tolerance
        and min(start.y, end.y) - tolerance <= point.y <= max(start.y, end.y) + tolerance
    )


def segments_intersect(
    first_start: Coordinate,
    first_end: Coordinate,
    second_start: Coordinate,
    second_end: Coordinate,
) -> bool:
    """Detect crossings or overlaps, excluding an already shared endpoint."""
    first_side = _cross(first_start, first_end, second_start)
    second_side = _cross(first_start, first_end, second_end)
    third_side = _cross(second_start, second_end, first_start)
    fourth_side = _cross(second_start, second_end, first_end)
    if (
        (first_side > 0 > second_side or first_side < 0 < second_side)
        and (third_side > 0 > fourth_side or third_side < 0 < fourth_side)
    ):
        return True
    contacts = [
        point
        for point, start, end in (
            (second_start, first_start, first_end),
            (second_end, first_start, first_end),
            (first_start, second_start, second_end),
            (first_end, second_start, second_end),
        )
        if point_on_segment(point, start, end)
    ]
    unique_contacts: list[Coordinate] = []
    for contact in contacts:
        if not any(distance(contact, existing) <= 1e-6 for existing in unique_contacts):
            unique_contacts.append(contact)
    if len(unique_contacts) > 1:
        return True
    if not unique_contacts:
        return False
    contact = unique_contacts[0]
    shared_endpoints = [
        first
        for first in (first_start, first_end)
        for second in (second_start, second_end)
        if distance(first, second) <= 1e-6
    ]
    return not any(distance(contact, shared) <= 1e-6 for shared in shared_endpoints)


def point_in_polygon(point: Coordinate, polygon: list[Coordinate]) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    if any(point_on_segment(point, start, end) for start, end in edges):
        return True
    inside = False
    for start, end in edges:
        if (start.y > point.y) == (end.y > point.y):
            continue
        crossing_x = (
            start.x
            + (point.y - start.y) * (end.x - start.x) / (end.y - start.y)
        )
        if point.x < crossing_x:
            inside = not inside
    return inside


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
