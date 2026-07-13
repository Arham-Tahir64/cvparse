"""Shared geometry helpers used across pipeline modules."""
from __future__ import annotations

import math

import numpy as np

from .models import LineSegment, Point


def angle_diff_rad(a: float, b: float) -> float:
    """Smallest difference between two direction-independent angles in [0, pi)."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def unit_direction(seg: LineSegment) -> tuple[float, float]:
    dx = seg.end.x - seg.start.x
    dy = seg.end.y - seg.start.y
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def project_param(point: Point, seg_start: Point, seg_end: Point) -> float:
    """Parameter t of the projection of `point` onto the segment line (t in R)."""
    dx = seg_end.x - seg_start.x
    dy = seg_end.y - seg_start.y
    denom = dx * dx + dy * dy
    if denom == 0:
        return 0.0
    return ((point.x - seg_start.x) * dx + (point.y - seg_start.y) * dy) / denom


def point_at_param(seg_start: Point, seg_end: Point, t: float) -> Point:
    return Point(
        seg_start.x + t * (seg_end.x - seg_start.x),
        seg_start.y + t * (seg_end.y - seg_start.y),
    )


def point_to_segment_distance(point: Point, seg_start: Point, seg_end: Point) -> float:
    t = max(0.0, min(1.0, project_param(point, seg_start, seg_end)))
    proj = point_at_param(seg_start, seg_end, t)
    return point.distance_to(proj)


def point_to_line_distance(point: Point, seg_start: Point, seg_end: Point) -> float:
    """Perpendicular distance to the infinite line through the segment."""
    proj = point_at_param(seg_start, seg_end, project_param(point, seg_start, seg_end))
    return point.distance_to(proj)


def sample_line_pixels(
    image: np.ndarray, p0: Point, p1: Point, num_samples: int | None = None
) -> np.ndarray:
    """Sample pixel values along the segment p0->p1. Out-of-bounds samples skipped."""
    length = p0.distance_to(p1)
    n = num_samples if num_samples is not None else max(2, int(round(length)))
    ts = np.linspace(0.0, 1.0, n)
    xs = np.rint(p0.x + ts * (p1.x - p0.x)).astype(int)
    ys = np.rint(p0.y + ts * (p1.y - p0.y)).astype(int)
    h, w = image.shape[:2]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return image[ys[valid], xs[valid]]


def liang_barsky_intersects(
    p0: Point, p1: Point, rect: tuple[float, float, float, float]
) -> bool:
    """True if segment p0->p1 intersects the rectangle (x_min, y_min, x_max, y_max)."""
    x_min, y_min, x_max, y_max = rect
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, p0.x - x_min),
        (dx, x_max - p0.x),
        (-dy, p0.y - y_min),
        (dy, y_max - p0.y),
    ):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def segment_orientation(seg: LineSegment) -> str:
    """Classify a segment as H, V, or diagonal from its angle."""
    if seg.is_horizontal:
        return "H"
    if seg.is_vertical:
        return "V"
    return "diagonal"


def overlap_interval(
    a_lo: float, a_hi: float, b_lo: float, b_hi: float
) -> tuple[float, float]:
    """Overlap of two 1D intervals; (lo, hi) with lo >= hi meaning no overlap."""
    return max(a_lo, b_lo), min(a_hi, b_hi)
