"""Tests for module 05 - wall extraction, against the spec's test criteria."""
import math

import numpy as np
import pytest

from vision.cv import wall_extraction
from vision.cv.config import PipelineConfig
from vision.cv.models import LineSegment, NoWallsFoundError, PipelineState, Point


def seg(x1, y1, x2, y2, thickness=1.5, sid="", cls="unknown"):
    return LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness,
                       classification=cls, id=sid or f"s{x1}_{y1}_{x2}_{y2}")


def make_state(lines, binary=None, config=None):
    state = PipelineState(config=config or PipelineConfig())
    state.classified_lines = list(lines)
    state.binary = binary if binary is not None else np.zeros((600, 600), np.uint8)
    state.image = np.full((600, 600), 255, np.uint8)
    state.structural_roi_mask = np.full((600, 600), 255, np.uint8)
    return state


def filled_band(y0, y1, x0, x1, shape=(600, 600)):
    binary = np.zeros(shape, np.uint8)
    binary[y0:y1 + 1, x0:x1 + 1] = 255
    return binary


def test_paired_faces_wall_with_support():
    binary = filled_band(100, 108, 100, 400)
    lines = [seg(100, 100, 400, 100, sid="a"), seg(100, 108, 400, 108, sid="b")]
    state = make_state(lines, binary)
    wall_extraction.run(state)
    paired = [w for w in state.walls if w.merge_kind == "paired_faces"]
    assert len(paired) == 1
    wall = paired[0]
    assert wall.fit_support_ratio > 0.8
    assert set(wall.source_ids) == {"a", "b"}
    assert abs(wall.centerline.midpoint.y - 104) < 2
    assert abs(wall.thickness - 8) < 1.5


def test_too_far_apart_no_pair():
    lines = [seg(100, 100, 400, 100), seg(100, 200, 400, 200)]  # 100 px apart
    state = make_state(lines, filled_band(100, 200, 100, 400))
    config = state.config
    with pytest.raises(NoWallsFoundError):
        # also below: disable single-face fallback so nothing else fires
        config.single_face_min_thickness_px = math.inf
        config.thin_branch_min_thickness_px = math.inf
        wall_extraction.run(state)


def test_too_close_no_pair():
    lines = [seg(100, 100, 400, 100), seg(100, 101, 400, 101)]  # 1 px apart
    state = make_state(lines, filled_band(99, 102, 100, 400))
    state.config.single_face_min_thickness_px = math.inf
    state.config.thin_branch_min_thickness_px = math.inf
    with pytest.raises(NoWallsFoundError):
        wall_extraction.run(state)


def test_low_overlap_no_pair():
    lines = [seg(100, 100, 400, 100), seg(385, 108, 700, 108)]  # ~5% overlap
    state = make_state(lines, filled_band(100, 108, 100, 600))
    state.config.single_face_min_thickness_px = math.inf
    state.config.thin_branch_min_thickness_px = math.inf
    with pytest.raises(NoWallsFoundError):
        wall_extraction.run(state)


def test_empty_between_rejected_by_face_support():
    binary = np.zeros((600, 600), np.uint8)  # no ink anywhere
    lines = [seg(100, 100, 400, 100), seg(100, 108, 400, 108)]
    state = make_state(lines, binary)
    state.config.single_face_min_thickness_px = math.inf
    state.config.thin_branch_min_thickness_px = math.inf
    with pytest.raises(NoWallsFoundError):
        wall_extraction.run(state)


def test_greedy_matching_prefers_higher_overlap():
    binary = filled_band(100, 108, 100, 400)
    binary[150:159, 100:400] = 255
    a = seg(100, 100, 400, 100, sid="A")
    b = seg(100, 108, 400, 108, sid="B")      # full overlap with A
    c = seg(250, 116, 550, 116, sid="C")      # partial overlap with B
    binary[108:117, 250:400] = 255
    state = make_state([a, b, c], binary)
    wall_extraction.run(state)
    paired = [w for w in state.walls if w.merge_kind == "paired_faces"]
    assert len(paired) == 1
    assert set(paired[0].source_ids) == {"A", "B"}


def test_centerline_equidistant():
    binary = filled_band(200, 210, 100, 500)
    lines = [seg(100, 200, 500, 200, sid="a"), seg(100, 210, 500, 210, sid="b")]
    state = make_state(lines, binary)
    wall_extraction.run(state)
    wall = [w for w in state.walls if w.merge_kind == "paired_faces"][0]
    assert abs(wall.centerline.midpoint.y - 205) <= 1.0


def test_thin_wall_recovery_with_orthogonal_support():
    # primary wall: vertical pair
    binary = filled_band(100, 400, 100, 108).T.copy()
    binary = np.zeros((600, 600), np.uint8)
    binary[100:400, 100:109] = 255           # vertical wall body
    binary[200:205, 109:175] = 255           # thin partition heading right
    primary = [
        seg(100, 100, 100, 400, sid="v1"),
        seg(108, 100, 108, 400, sid="v2"),
    ]
    # 60 px partition attached at the wall; midpoint within 40 px of the wall
    thin = seg(110, 202, 170, 202, thickness=4.0, sid="thin")
    state = make_state(primary + [thin], binary)
    state.config.single_face_min_thickness_px = math.inf
    wall_extraction.run(state)
    thin_walls = [w for w in state.walls if "thin" in w.source_ids]
    assert len(thin_walls) == 1
    assert thin_walls[0].merge_kind == "single_face"
    assert thin_walls[0].fit_support_ratio == 1.0
    assert thin_walls[0].merge_confidence == 1.0


def test_thin_candidate_without_orthogonal_support_rejected():
    binary = np.zeros((600, 600), np.uint8)
    binary[100:400, 100:109] = 255
    binary[500:505, 400:500] = 255
    primary = [seg(100, 100, 100, 400, sid="v1"), seg(108, 100, 108, 400, sid="v2")]
    # parallel to the primary wall's direction? no - make it horizontal but far away
    thin = seg(400, 502, 500, 502, thickness=4.0, sid="thin")  # 100px, far from walls
    state = make_state(primary + [thin], binary)
    state.config.single_face_min_thickness_px = math.inf
    wall_extraction.run(state)
    assert not any("thin" in w.source_ids for w in state.walls)


def test_single_face_fallback_provenance():
    binary = np.zeros((600, 600), np.uint8)
    binary[300:306, 100:500] = 255
    lines = [seg(100, 302, 500, 302, thickness=6.0, sid="lone")]
    state = make_state(lines, binary)
    wall_extraction.run(state)
    assert len(state.walls) == 1
    wall = state.walls[0]
    assert wall.merge_kind == "single_face"
    assert wall.fit_support_ratio == 1.0
    assert wall.merge_confidence == 1.0
    assert wall.source_ids == ["lone"]


def test_manhattan_snap_2deg():
    binary = np.zeros((600, 600), np.uint8)
    binary[298:312, 100:500] = 255
    dy = math.tan(math.radians(2)) * 400
    lines = [seg(100, 300, 500, 300 + dy, thickness=6.0, sid="a")]
    state = make_state(lines, binary)
    wall_extraction.run(state)
    wall = state.walls[0]
    assert wall.orientation == "H"
    assert abs(wall.centerline.start.y - wall.centerline.end.y) < 1e-6


def test_30deg_not_snapped():
    binary = np.zeros((600, 600), np.uint8)
    lines = [seg(100, 100, 400, 100 + math.tan(math.radians(30)) * 300,
                 thickness=6.0, sid="a")]
    # draw diagonal band
    import cv2
    cv2.line(binary, (100, 100), (400, int(100 + math.tan(math.radians(30)) * 300)), 255, 8)
    state = make_state(lines, binary)
    wall_extraction.run(state)
    wall = state.walls[0]
    assert wall.orientation == "diagonal"


def test_visual_thickness_populated():
    binary = filled_band(100, 108, 100, 400)
    lines = [seg(100, 100, 400, 100, sid="a"), seg(100, 108, 400, 108, sid="b")]
    state = make_state(lines, binary)
    wall_extraction.run(state)
    for wall in state.walls:
        assert 0 < wall.visual_thickness <= state.config.visual_thickness_max_px
    paired = [w for w in state.walls if w.merge_kind == "paired_faces"][0]
    assert 6 <= paired.visual_thickness <= 12


def test_no_walls_raises():
    state = make_state([], np.zeros((600, 600), np.uint8))
    with pytest.raises(NoWallsFoundError):
        wall_extraction.run(state)
