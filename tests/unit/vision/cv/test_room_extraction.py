"""Tests for module 09 - room extraction, against the spec's test criteria."""
import cv2
import numpy as np
import pytest

from vision.cv import junction_snapping, room_extraction
from vision.cv.config import PipelineConfig
from vision.cv.models import (
    Junction, LineSegment, NoRoomsExtractedError, PipelineState, Point,
    TextElement, Wall,
)


def wall(wid, x1, y1, x2, y2, thickness=6.0):
    cl = LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness)
    return Wall(
        id=wid, orientation="H" if abs(y2 - y1) < abs(x2 - x1) else "V",
        centerline=cl, thickness=thickness, visual_thickness=thickness,
        merge_kind="paired_faces", fit_support_ratio=0.9, merge_confidence=0.85,
        source_ids=[], length_px=cl.length,
    )


def make_state(walls, shape=(600, 800)):
    state = PipelineState(config=PipelineConfig())
    state.walls = list(walls)
    state.image = np.full(shape, 255, np.uint8)
    state.binary = np.zeros(shape, np.uint8)
    state.binary_masked = state.binary.copy()
    # produce junctions via module 06 so the graph has real topology
    junction_snapping.run(state)
    return state


def square_walls():
    return [
        wall("W0001", 100, 100, 500, 100),
        wall("W0002", 500, 100, 500, 400),
        wall("W0003", 500, 400, 100, 400),
        wall("W0004", 100, 400, 100, 100),
    ]


def test_square_one_room():
    state = make_state(square_walls())
    room_extraction.run(state)
    assert len(state.rooms) == 1
    assert abs(state.rooms[0].area - 400 * 300) < 0.05 * 400 * 300


def test_two_adjacent_rooms():
    walls = square_walls() + [wall("W0005", 300, 100, 300, 400)]
    state = make_state(walls)
    room_extraction.run(state)
    assert len(state.rooms) == 2


def test_l_shaped_room():
    walls = [
        wall("W1", 100, 100, 500, 100),
        wall("W2", 500, 100, 500, 250),
        wall("W3", 500, 250, 300, 250),
        wall("W4", 300, 250, 300, 450),
        wall("W5", 300, 450, 100, 450),
        wall("W6", 100, 450, 100, 100),
    ]
    state = make_state(walls)
    room_extraction.run(state)
    assert len(state.rooms) == 1
    expected = 400 * 150 + 200 * 200
    assert abs(state.rooms[0].area - expected) < 0.05 * expected


def test_outer_face_absent():
    state = make_state(square_walls())
    room_extraction.run(state)
    image_area = 600 * 800
    for room in state.rooms:
        assert room.area < 0.85 * image_area


def test_small_face_discarded():
    walls = square_walls() + [
        # a 20x30 closet in the corner: 600 px^2 < min_room_area_px (1000)
        wall("Wa", 100, 100, 120, 100),
        wall("Wb", 120, 100, 120, 130),
        wall("Wc", 120, 130, 100, 130),
    ]
    state = make_state(walls)
    room_extraction.run(state)
    assert all(r.area >= state.config.min_room_area_px for r in state.rooms)


def test_door_passage_junction_ok():
    state = make_state(square_walls())
    state.junctions.append(Junction(
        id="J9999", point=Point(300, 100), walls=["W0001"],
        junction_type="door_passage",
    ))
    room_extraction.run(state)
    assert len(state.rooms) >= 1


def test_crossing_walls_fall_back_and_extract():
    # X-crossing inside a square: non-planar embedding data (shared endpoints
    # resolve, but the diagonals cross mid-air without a junction)
    walls = square_walls() + [
        wall("W0005", 100, 100, 500, 400),
        wall("W0006", 500, 100, 100, 400),
    ]
    state = make_state(walls)
    room_extraction.run(state)
    assert len(state.rooms) >= 1


def test_dilation_closes_small_gap():
    walls = [
        wall("W0001", 100, 100, 497, 100),  # 3 px short of the corner
        wall("W0002", 500, 100, 500, 400),
        wall("W0003", 500, 400, 100, 400),
        wall("W0004", 100, 400, 100, 100),
    ]
    state = PipelineState(config=PipelineConfig())
    state.walls = walls
    state.image = np.full((600, 800), 255, np.uint8)
    state.binary = np.zeros((600, 800), np.uint8)
    state.junctions = []
    rooms = room_extraction._floodfill_rooms(state, 600 * 800, __import__(
        "vision.cv.models", fromlist=["IdGenerator"]).IdGenerator("R"))
    assert len(rooms) == 1


def test_two_disconnected_enclosures():
    walls = square_walls() + [
        wall("Wa", 550, 100, 750, 100),
        wall("Wb", 750, 100, 750, 300),
        wall("Wc", 750, 300, 550, 300),
        wall("Wd", 550, 300, 550, 100),
    ]
    state = make_state(walls)
    room_extraction.run(state)
    assert len(state.rooms) == 2


def test_no_rooms_raises():
    walls = [wall("W0001", 100, 100, 500, 100)]  # single open wall
    state = make_state(walls)
    with pytest.raises(NoRoomsExtractedError):
        room_extraction.run(state)


def test_polygon_first_vertex_not_repeated():
    state = make_state(square_walls())
    room_extraction.run(state)
    poly = state.rooms[0].polygon
    assert poly[0].distance_to(poly[-1]) > 1.0


def test_semantic_raster_rooms_bridge_openings_and_keep_labels():
    state = PipelineState(config=PipelineConfig(
        room_barrier_min_line_px=12,
        room_barrier_gap_close_px=35,
        room_barrier_thickness_px=5,
    ))
    state.image = np.full((300, 500), 255, np.uint8)
    state.binary = np.zeros((300, 500), np.uint8)
    # Outer enclosure and a partition with a door-sized interruption.
    cv2 = __import__("cv2")
    cv2.rectangle(state.binary, (40, 40), (460, 260), 255, 6)
    cv2.line(state.binary, (250, 40), (250, 130), 255, 6)
    cv2.line(state.binary, (250, 160), (250, 260), 255, 6)
    state.binary_masked = state.binary.copy()
    state.structural_core_mask = np.zeros_like(state.binary)
    state.structural_core_mask[25:276, 25:476] = 255
    state.raw_texts = [
        TextElement("GUEST SUITE", (100, 130, 180, 150), 0.95),
        TextElement("BATH", (330, 130, 380, 150), 0.93),
    ]

    room_extraction.run(state)

    assert len(state.rooms) == 2
    assert {room.label for room in state.rooms} == {"GUEST SUITE", "BATH"}
    assert all(len(room.polygon) >= 4 for room in state.rooms)


def test_room_clip_rectangularizes_only_high_fill_manhattan_envelope():
    state = PipelineState(config=PipelineConfig(
        semantic_plan_margin_px=0,
        semantic_plan_rectangularize_min_fill=0.90,
    ))
    state.image = np.full((300, 400), 255, np.uint8)
    mask = np.zeros(state.image.shape, np.uint8)
    cv2 = __import__("cv2")
    cv2.fillPoly(mask, [np.array([
        [50, 50], [350, 50], [350, 250], [320, 250], [50, 220],
    ], np.int32)], 255)
    mask = room_extraction._rectangularized_room_clip(state, mask)

    assert mask[245, 60] == 255

    mask = np.zeros(state.image.shape, np.uint8)
    cv2.fillPoly(mask, [np.array([
        [50, 50], [200, 50], [200, 150],
        [350, 150], [350, 250], [50, 250],
    ], np.int32)], 255)
    mask = room_extraction._rectangularized_room_clip(state, mask)

    assert mask[80, 300] == 0


def test_obstructed_semantic_seed_snaps_to_nearest_free_component():
    labels = np.zeros((40, 60), np.int32)
    labels[8:32, 12:50] = 7
    # Simulate OCR text/hatch ink covering the exact centre of the label.
    labels[18:23, 28:33] = 0

    snapped = room_extraction._nearest_free_seed(labels, 30, 20, radius=8)

    assert snapped is not None
    component, x, y = snapped
    assert component == 7
    assert labels[y, x] == 7
    assert (x - 30) ** 2 + (y - 20) ** 2 <= 8 ** 2


def test_manhattan_semantic_partition_preserves_component_and_axis_boundary():
    component = np.zeros((80, 100), np.uint8)
    component[10:70, 10:90] = 255
    seeds = [
        ("STAIR/CIRCULATION", 1.0, 45, 25),
        ("REC ROOM AREA", 0.95, 55, 60),
    ]

    partitions = room_extraction._partition_seeded_component(
        component, seeds, manhattan=True,
    )

    assert len(partitions) == 2
    upper = partitions[0][2]
    lower = partitions[1][2]
    assert upper[20, 50] == 255
    assert upper[60, 50] == 0
    assert lower[20, 50] == 0
    assert lower[60, 50] == 255
    assert np.count_nonzero(cv2.bitwise_and(upper, lower)) == 0
    assert np.array_equal(cv2.bitwise_or(upper, lower), component)
