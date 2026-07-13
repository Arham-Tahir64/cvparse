"""Tests for module 09 - room extraction, against the spec's test criteria."""
import numpy as np
import pytest

from vision.cv import junction_snapping, room_extraction
from vision.cv.config import PipelineConfig
from vision.cv.models import (
    Junction, LineSegment, NoRoomsExtractedError, PipelineState, Point, Wall,
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
