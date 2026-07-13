"""Tests for independent filled wall/window/room masks."""
import numpy as np

from vision.cv import semantic_masks
from vision.cv.config import PipelineConfig
from vision.cv.models import Door, LineSegment, PipelineState, Point, Room, Wall, Window


def wall(wid, x1, y1, x2, y2, thickness=16):
    line = LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness)
    return Wall(
        id=wid, orientation="H" if y1 == y2 else "V", centerline=line,
        thickness=thickness, visual_thickness=thickness,
        merge_kind="paired_faces", fit_support_ratio=1.0,
        merge_confidence=1.0, source_ids=[], length_px=line.length,
    )


def test_windows_are_full_spans_cut_out_of_supporting_walls():
    state = PipelineState(config=PipelineConfig())
    state.image = np.full((300, 400), 255, np.uint8)
    state.walls = [wall("WH", 50, 60, 350, 60), wall("WV", 300, 80, 300, 260)]
    state.windows = [
        Window("WD1", Point(180, 60), 80, "WH"),
        Window("WD2", Point(300, 170), 70, "WV"),
    ]

    semantic_masks.run(state)

    assert state.window_mask[60, 180] == 255
    assert state.window_mask[170, 300] == 255
    assert state.window_mask[60, 180] != state.wall_mask[60, 180]
    assert state.wall_mask[60, 80] == 255
    # Vertical window span follows the vertical supporting wall.
    assert state.window_mask[140, 300] == 255
    assert state.window_mask[170, 260] == 0
    assert tuple(state.combined_class_mask[60, 180]) == semantic_masks.CLASS_COLORS["window"]
    assert tuple(state.combined_class_mask[60, 80]) == semantic_masks.CLASS_COLORS["wall"]


def test_room_and_door_masks_are_exported_separately():
    state = PipelineState(config=PipelineConfig())
    state.image = np.full((300, 400), 255, np.uint8)
    state.rooms = [Room(
        "R1", [Point(40, 40), Point(360, 40), Point(360, 260), Point(40, 260)],
        label="ROOM", area=70400,
    )]
    state.doors = [Door(
        "D1", Point(100, 100), Point(100, 150), 50, "W1", "cw",
        swing_arc=[Point(150, 100), Point(135, 135), Point(100, 150)],
    )]

    semantic_masks.run(state)

    assert state.room_region_mask[200, 200] == 255
    assert state.door_mask[120, 120] == 255
    assert tuple(state.combined_class_mask[120, 120]) == semantic_masks.CLASS_COLORS["door"]
