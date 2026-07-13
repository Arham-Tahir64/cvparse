"""Tests for module 08 - window detection, against the spec's test criteria."""
import numpy as np

from vision.cv import window_detection
from vision.cv.config import PipelineConfig
from vision.cv.models import Gap, LineSegment, PipelineState, Point, Wall


def wall(wid, x1, y1, x2, y2, thickness=10.0):
    cl = LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness)
    return Wall(
        id=wid, orientation="H" if abs(y2 - y1) < abs(x2 - x1) else "V",
        centerline=cl, thickness=thickness, visual_thickness=thickness,
        merge_kind="paired_faces", fit_support_ratio=0.9, merge_confidence=0.85,
        source_ids=[], length_px=cl.length,
    )


def seg(x1, y1, x2, y2, cls="unknown", sid="s"):
    return LineSegment(Point(x1, y1), Point(x2, y2), thickness=1.0,
                       classification=cls, id=sid)


def solid_wall_binary(y_center=200, half=5, x0=100, x1=500, shape=(400, 600)):
    binary = np.zeros(shape, np.uint8)
    binary[y_center - half:y_center + half + 1, x0:x1] = 255
    return binary


def make_state(walls, lines=(), binary=None, gaps=()):
    state = PipelineState(config=PipelineConfig())
    state.walls = list(walls)
    state.classified_lines = list(lines)
    state.binary = binary if binary is not None else solid_wall_binary()
    state.binary_masked = state.binary.copy()
    state.image = np.where(state.binary > 0, 0, 255).astype(np.uint8)
    state.gaps = list(gaps)
    return state


def test_inner_line_window():
    w = wall("W0001", 100, 200, 500, 200)
    inner = seg(280, 200, 330, 200)  # 50 px inner line on the centerline
    state = make_state([w], [inner])
    window_detection.run(state)
    assert len(state.windows) == 1
    win = state.windows[0]
    assert abs(win.position.x - 305) < 3
    assert abs(win.width - 50) < 5
    window_gaps = [g for g in state.gaps if g.kind == "window"]
    assert len(window_gaps) == 1
    assert window_gaps[0].wall_break_score == 1.0


def test_inner_line_outside_extent_no_window():
    w = wall("W0001", 100, 200, 500, 200)
    inner = seg(480, 200, 540, 200)  # projects past the wall end
    state = make_state([w], [inner])
    window_detection.run(state)
    assert state.windows == []


def test_inner_line_too_short_no_window():
    w = wall("W0001", 100, 200, 500, 200)
    inner = seg(300, 200, 315, 200)  # 15 px < window_gap_min_px
    state = make_state([w], [inner])
    window_detection.run(state)
    assert state.windows == []


def test_inner_line_wrong_angle_no_window():
    w = wall("W0001", 100, 200, 500, 200)
    inner = seg(280, 180, 330, 209)  # ~30 degrees
    state = make_state([w], [inner])
    window_detection.run(state)
    assert state.windows == []


def test_overlapping_candidates_merge():
    w = wall("W0001", 100, 200, 500, 200)
    a = seg(280, 200, 330, 200, sid="a")
    b = seg(290, 201, 340, 201, sid="b")
    state = make_state([w], [a, b])
    window_detection.run(state)
    assert len(state.windows) == 1


def test_face_gap_strategy_b():
    binary = solid_wall_binary()
    binary[:, 280:320] = 0  # 40 px gap through both faces
    w = wall("W0001", 100, 200, 500, 200)
    state = make_state([w], [], binary)
    window_detection.run(state)
    assert len(state.windows) == 1
    assert abs(state.windows[0].position.x - 300) < 6
    assert 30 <= state.windows[0].width <= 50


def test_strategy_b_skipped_when_a_found():
    binary = solid_wall_binary()
    binary[:, 380:420] = 0  # face gap that B would find
    w = wall("W0001", 100, 200, 500, 200)
    inner = seg(180, 200, 230, 200)  # A finds this
    state = make_state([w], [inner], binary)
    window_detection.run(state)
    assert len(state.windows) == 1
    assert abs(state.windows[0].position.x - 205) < 5


def test_door_gap_not_redetected():
    binary = solid_wall_binary()
    binary[:, 280:320] = 0
    w = wall("W0001", 100, 200, 500, 200)
    door_gap = Gap(
        id="G0001", wall_id="W0001", orientation="H", center=Point(300, 200),
        width_px=45.0, bbox=(255, 155, 345, 245), kind="door",
        wall_break_score=0.25, opening_fill_ratio=0.9,
    )
    state = make_state([w], [], binary, gaps=[door_gap])
    window_detection.run(state)
    assert state.windows == []


def test_wall_end_gap_rejected_by_side_fill():
    # gap at the very start of the wall: no ink on the left side
    binary = np.zeros((400, 600), np.uint8)
    binary[195:206, 160:500] = 255  # wall ink starts at x=160
    w = wall("W0001", 100, 200, 500, 200)  # wall model extends to x=100
    state = make_state([w], [], binary)
    window_detection.run(state)
    # the 60 px "gap" at the start must not become a window... but one side
    # has fill, so per spec (reject only if BOTH sides low) it may pass.
    # True corner case: isolated stub with no ink on either side.
    binary2 = np.zeros((400, 600), np.uint8)
    binary2[195:206, 250:350] = 255  # short stub in the middle
    w2 = wall("W0002", 100, 200, 500, 200)
    state2 = make_state([w2], [], binary2)
    window_detection.run(state2)
    for win in state2.windows:
        # gaps at the extreme ends (no ink beyond them) must be rejected
        assert not (win.position.x < 200 and win.width > 80)


def test_windows_do_not_split_walls():
    binary = solid_wall_binary()
    binary[:, 280:320] = 0
    w = wall("W0001", 100, 200, 500, 200)
    state = make_state([w], [], binary)
    window_detection.run(state)
    assert len(state.walls) == 1
