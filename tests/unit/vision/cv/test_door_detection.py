"""Tests for module 07 - door detection, against the spec's test criteria."""
import math

import cv2
import numpy as np

from vision.cv import door_detection
from vision.cv.config import PipelineConfig
from vision.cv.models import LineSegment, PipelineState, Point, Wall


def wall(wid, x1, y1, x2, y2, thickness=8.0):
    cl = LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness)
    return Wall(
        id=wid, orientation="H" if abs(y2 - y1) < abs(x2 - x1) else "V",
        centerline=cl, thickness=thickness, visual_thickness=thickness,
        merge_kind="paired_faces", fit_support_ratio=0.9, merge_confidence=0.85,
        source_ids=[f"{wid}-src"], length_px=cl.length,
    )


def make_state(walls, binary):
    # param2=25: thin synthetic quarter arcs need a lower accumulator
    # threshold than the field default (spec: tune 20-50)
    state = PipelineState(config=PipelineConfig(
        hough_circles_param2=25.0, door_arc_min_radius_px=20.0,
    ))
    state.walls = list(walls)
    state.binary = binary
    state.binary_masked = binary.copy()
    state.image = np.where(binary > 0, 0, 255).astype(np.uint8)
    return state


def draw_arc(binary, cx, cy, radius, start_deg, end_deg, thickness=2):
    cv2.ellipse(binary, (cx, cy), (radius, radius), 0, start_deg, end_deg,
                255, thickness)
    angle = math.radians(end_deg)
    cv2.line(binary, (cx, cy),
             (round(cx + radius * math.cos(angle)),
              round(cy + radius * math.sin(angle))), 255, thickness)


def open_wall(binary, cx, radius, y=200, thickness=9):
    binary[y - thickness // 2:y + thickness // 2 + 1, cx:cx + radius + 2] = 0


def test_quarter_arc_near_wall_detected():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255           # horizontal wall band
    open_wall(binary, 300, 50)
    draw_arc(binary, 300, 200, 50, 0, 90)   # hinge on the wall, opens south
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert len(state.doors) == 1
    door = state.doors[0]
    assert door.wall_id in [w.id for w in state.walls] or door.wall_id == "W0001"
    assert abs(door.position.y - 200) < 3
    assert 40 <= door.radius <= 60


def test_common_scaled_door_radius_is_within_search_defaults():
    """A 2'-6" leaf at 1/4" scale and 200 DPI is about 125 pixels."""
    config = PipelineConfig()
    assert config.door_arc_min_radius_px < 125 < config.door_arc_max_radius_px
    assert config.hough_circles_param2 <= 25


def test_thin_paired_walls_adapt_door_proposal_radius_downward():
    state = PipelineState(config=PipelineConfig())
    state.walls = [wall("W1", 0, 0, 300, 0, thickness=16)]

    minimum, maximum = door_detection._adaptive_arc_radius_bounds(
        state, state.config,
    )

    assert minimum == 40
    assert maximum == state.config.door_arc_max_radius_px


def test_thick_paired_walls_keep_nominal_door_minimum():
    state = PipelineState(config=PipelineConfig())
    state.walls = [wall("W1", 0, 0, 500, 0, thickness=40)]

    minimum, _ = door_detection._adaptive_arc_radius_bounds(
        state, state.config,
    )

    assert minimum == state.config.door_arc_min_radius_px


def test_short_wall_fragments_do_not_set_door_scale():
    state = PipelineState(config=PipelineConfig())
    state.walls = [wall("W1", 0, 0, 30, 0, thickness=20)]

    minimum, _ = door_detection._adaptive_arc_radius_bounds(
        state, state.config,
    )

    assert minimum == state.config.door_arc_min_radius_px


def test_circle_center_must_remain_within_one_radius_of_snapped_hinge():
    config = PipelineConfig()
    hinge = Point(100, 100)

    assert door_detection._hinge_center_offset_valid(
        Point(100, 140), hinge, 50, config.door_max_hinge_offset_ratio,
    )
    assert not door_detection._hinge_center_offset_valid(
        Point(100, 160), hinge, 50, config.door_max_hinge_offset_ratio,
    )


def test_full_circle_rejected():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    cv2.circle(binary, (300, 260), 50, 255, 2)
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert state.doors == []


def test_fixture_arc_on_uninterrupted_wall_rejected():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    draw_arc(binary, 300, 200, 50, 0, 90)
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert state.doors == []


def test_arc_far_from_walls_rejected():
    binary = np.zeros((500, 600), np.uint8)
    binary[46:55, 50:550] = 255
    draw_arc(binary, 300, 400, 50, 0, 90)   # 350 px from the wall
    state = make_state([wall("W0001", 50, 50, 550, 50)], binary)
    door_detection.run(state)
    assert state.doors == []


def test_swing_direction_south_arc():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    open_wall(binary, 300, 50)
    draw_arc(binary, 300, 200, 50, 0, 90)   # south of a horizontal wall
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert len(state.doors) == 1
    # wall direction +x, swing vector points south (+y) -> cross > 0 -> cw
    assert state.doors[0].swing_direction == "cw"


def test_two_close_arcs_one_door():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    open_wall(binary, 300, 55)
    draw_arc(binary, 300, 200, 50, 10, 90)
    draw_arc(binary, 305, 200, 45, 10, 90)
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert len(state.doors) == 1


def test_wall_split_at_hinge():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 100:500] = 255
    open_wall(binary, 260, 50)
    draw_arc(binary, 260, 200, 50, 0, 90)   # t = 0.4 along a 400 px wall
    state = make_state([wall("W0001", 100, 200, 500, 200)], binary)
    door_detection.run(state)
    assert len(state.doors) == 1
    assert len(state.walls) == 2
    lengths = sorted(w.centerline.length for w in state.walls)
    assert abs(sum(lengths) - 400) < 10
    passage = [j for j in state.junctions if j.junction_type == "door_passage"]
    assert len(passage) == 1


def test_no_circles_no_doors_no_error():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert state.doors == []


def test_diagonal_sector_snaps_to_existing_cardinal_quadrant():
    closed, leaf = door_detection._snap_sector_to_cardinal_quadrant(
        math.radians(225), math.radians(135),
    )

    assert math.degrees(closed) == 180
    assert math.degrees(leaf) == 90


def test_split_children_carry_provenance():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 100:500] = 255
    open_wall(binary, 260, 50)
    draw_arc(binary, 260, 200, 50, 0, 90)
    state = make_state([wall("W0001", 100, 200, 500, 200)], binary)
    door_detection.run(state)
    assert len(state.walls) == 2
    for w in state.walls:
        assert w.fit_support_ratio == 0.9
        assert w.merge_confidence == 0.85
        assert "W0001" in w.source_ids


def test_gap_record_created():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    open_wall(binary, 300, 50)
    draw_arc(binary, 300, 200, 50, 0, 90)
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    door_gaps = [g for g in state.gaps if g.kind == "door"]
    assert len(door_gaps) == 1
    assert door_gaps[0].wall_id == "W0001"
    assert 0 < door_gaps[0].wall_break_score <= 0.4 + 0.05


def test_detected_door_exports_ordered_swing_arc():
    binary = np.zeros((400, 600), np.uint8)
    binary[196:205, 50:550] = 255
    open_wall(binary, 300, 50)
    draw_arc(binary, 300, 200, 50, 0, 90)
    state = make_state([wall("W0001", 50, 200, 550, 200)], binary)
    door_detection.run(state)
    assert len(state.doors) == 1
    assert len(state.doors[0].swing_arc) >= 12
    assert state.doors[0].confidence > 0


def test_original_leaf_evidence_survives_structural_cleanup():
    source = np.zeros((400, 600), np.uint8)
    source[196:205, 50:550] = 255
    open_wall(source, 300, 50)
    draw_arc(source, 300, 200, 50, 0, 90)
    cleaned = source.copy()
    # Cleanup removed the radial leaf but retained enough swing arc for Hough.
    cleaned[205:248, 297:304] = 0
    state = make_state([wall("W0001", 50, 200, 550, 200)], source)
    state.binary_cleaned = cleaned

    door_detection.run(state)

    assert len(state.doors) == 1
    assert state.doors[0].confidence > 0


def _endpoint_repair_state(same_room=False):
    binary = np.zeros((400, 600), np.uint8)
    binary[191:210, 50:251] = 255
    binary[191:210, 300:551] = 255
    # Only half of the swing survives, so this is endpoint-repair evidence
    # rather than a normal complete quarter-circle proposal.
    cv2.ellipse(binary, (250, 200), (50, 50), 0, 0, 45, 255, 2)
    cv2.line(binary, (250, 200), (250, 250), 255, 2)
    state = make_state([
        wall("W0001", 50, 200, 250, 200, thickness=19),
        wall("W0002", 300, 200, 550, 200, thickness=19),
    ], binary)
    state.config = PipelineConfig(
        door_arc_min_radius_px=20.0,
        door_arc_max_radius_px=60.0,
    )
    state.semantic_plan_mask = np.full(binary.shape, 255, np.uint8)
    state.room_instance_mask = np.ones(binary.shape, np.int32)
    if not same_room:
        state.room_instance_mask[210:, :] = 2
    return state, binary


def test_endpoint_repair_requires_room_boundary_and_opposite_jamb():
    state, binary = _endpoint_repair_state()

    repairs = door_detection._endpoint_repair_candidates(
        state, binary, binary, [(255.0, 200.0, 50.0)], [],
    )

    assert len(repairs) == 1
    assert repairs[0][6].distance_to(Point(250, 200)) < 1
    assert repairs[0][1] == 50


def test_endpoint_repair_rejects_fixture_inside_one_room():
    state, binary = _endpoint_repair_state(same_room=True)

    repairs = door_detection._endpoint_repair_candidates(
        state, binary, binary, [(255.0, 200.0, 50.0)], [],
    )

    assert repairs == []
