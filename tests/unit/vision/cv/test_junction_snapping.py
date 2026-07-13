"""Tests for module 06 - junction snapping, against the spec's test criteria."""
import numpy as np

from vision.cv import junction_snapping
from vision.cv.config import PipelineConfig
from vision.cv.models import LineSegment, PipelineState, Point, Wall


def wall(wid, x1, y1, x2, y2, thickness=8.0):
    cl = LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness)
    return Wall(
        id=wid, orientation="H" if abs(y2 - y1) < abs(x2 - x1) else "V",
        centerline=cl, thickness=thickness, visual_thickness=thickness,
        merge_kind="paired_faces", fit_support_ratio=0.9, merge_confidence=0.8,
        source_ids=[f"{wid}-src"], length_px=cl.length,
    )


def make_state(walls):
    state = PipelineState(config=PipelineConfig())
    state.walls = list(walls)
    state.image = np.full((600, 600), 255, np.uint8)
    return state


def junction_types(state):
    return sorted(j.junction_type for j in state.junctions)


def test_two_walls_L_junction():
    state = make_state([
        wall("W0001", 100, 100, 300, 100),
        wall("W0002", 303, 102, 303, 300),  # endpoint 3.6 px away
    ])
    junction_snapping.run(state)
    l_junctions = [j for j in state.junctions if j.junction_type == "L"]
    assert len(l_junctions) == 1
    j = l_junctions[0]
    ends = [state.walls[0].centerline.end, ]
    for w in state.walls:
        cl = w.centerline
        assert min(cl.start.distance_to(j.point), cl.end.distance_to(j.point)) < 0.5 or True
    # both walls actually share the junction point
    touching = [w for w in state.walls
                if min(w.centerline.start.distance_to(j.point),
                       w.centerline.end.distance_to(j.point)) < 0.5]
    assert len(touching) == 2


def test_three_walls_T_junction():
    state = make_state([
        wall("W0001", 100, 200, 300, 200),
        wall("W0002", 302, 198, 500, 200),
        wall("W0003", 300, 203, 300, 400),
    ])
    junction_snapping.run(state)
    assert "T" in junction_types(state)


def test_four_walls_X_junction():
    state = make_state([
        wall("W0001", 100, 200, 300, 200),
        wall("W0002", 302, 200, 500, 200),
        wall("W0003", 300, 202, 300, 400),
        wall("W0004", 300, 198, 300, 50),
    ])
    junction_snapping.run(state)
    assert "X" in junction_types(state)


def test_isolated_endpoint_dead_end():
    state = make_state([wall("W0001", 100, 100, 300, 100)])
    junction_snapping.run(state)
    assert junction_types(state) == ["dead_end", "dead_end"]


def test_gap_closure_splits_target():
    state = make_state([
        wall("W0001", 100, 200, 500, 200),      # horizontal body
        wall("W0002", 300, 206, 300, 400),       # vertical end 6 px below body
    ])
    junction_snapping.run(state)
    assert "T" in junction_types(state)
    # horizontal wall split into two
    horizontals = [w for w in state.walls if w.orientation == "H"]
    assert len(horizontals) == 2
    for h in horizontals:
        assert "W0001" in h.source_ids


def test_gap_beyond_threshold_not_closed():
    state = make_state([
        wall("W0001", 100, 200, 500, 200),
        wall("W0002", 300, 220, 300, 400),  # 20 px gap
    ])
    junction_snapping.run(state)
    assert "T" not in junction_types(state)
    assert len(state.walls) == 2


def test_zero_length_wall_removed():
    state = make_state([
        wall("W0001", 100, 100, 300, 100),
        wall("W0002", 300, 100, 301, 100),  # collapses inside snap radius
        wall("W0003", 301, 100, 301, 300),
    ])
    junction_snapping.run(state)
    assert "W0002" not in [w.id for w in state.walls]


def test_invariant_all_endpoints_on_junctions():
    state = make_state([
        wall("W0001", 100, 200, 500, 200),
        wall("W0002", 300, 206, 300, 400),
        wall("W0003", 502, 198, 502, 400),
        wall("W0004", 100, 202, 100, 400),
    ])
    junction_snapping.run(state)
    points = [j.point for j in state.junctions]
    for w in state.walls:
        for endpoint in (w.centerline.start, w.centerline.end):
            assert min(endpoint.distance_to(p) for p in points) <= 0.5


def test_empty_input_no_error():
    state = make_state([])
    junction_snapping.run(state)
    assert state.junctions == []


def test_split_children_carry_source_ids():
    state = make_state([
        wall("W0001", 100, 200, 500, 200),
        wall("W0002", 250, 205, 250, 400),
    ])
    junction_snapping.run(state)
    children = [w for w in state.walls if "W0001" in w.source_ids and w.id != "W0001"]
    assert len(children) == 2
    for c in children:
        assert c.fit_support_ratio == 0.9
        assert c.merge_confidence == 0.8
        assert c.merge_kind == "paired_faces"
