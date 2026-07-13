"""Tests for independent filled wall/window/room masks."""
import cv2
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


def test_cleanup_protection_corridors_are_not_exported_as_semantic_walls():
    state = PipelineState(config=PipelineConfig(
        wall_region_axis_min_run_px=21,
    ))
    state.image = np.full((240, 420), 255, np.uint8)
    state.walls = [wall("W1", 40, 100, 150, 100, thickness=14)]
    state.structural_protection_mask = np.zeros(state.image.shape, np.uint8)
    cv2.line(state.structural_protection_mask, (40, 100), (380, 100), 255, 14)
    state.windows = [Window("WD1", Point(260, 100), 60, "W1")]

    semantic_masks.run(state)

    assert state.wall_boundary_mask is not None
    assert state.wall_polygon_mask[100, 100] == 255
    # Cleanup protection is intentionally permissive and must not manufacture
    # a semantic wall where the final clean pass found no wall.
    assert state.wall_repaired_mask[100, 350] == 0
    # Window ownership is subtracted after reconstruction and gap repair.
    assert state.wall_repaired_mask[100, 260] == 0
    assert state.window_mask[100, 260] == 255


def test_interior_width_uses_lower_consistent_face_pair_mode():
    state = PipelineState(config=PipelineConfig(
        wall_region_interior_width_quantile=0.40,
        wall_region_interior_width_scale=1.15,
    ))
    state.image = np.full((300, 500), 255, np.uint8)
    state.walls = [
        wall("W1", 30, 50, 470, 50, 10),
        wall("W2", 30, 100, 470, 100, 12),
        wall("W3", 30, 150, 470, 150, 14),
        wall("W4", 30, 230, 470, 230, 60),
    ]

    semantic_masks.run(state)

    limit = state.debug.segment_counts["13_interior_width_limit_px"]
    assert 13 <= limit <= 16
    assert state.wall_polygon_mask[230 + limit, 250] == 0
    assert state.wall_polygon_mask[230, 250] == 255


def test_room_boundary_support_keeps_thin_partition_and_rejects_floating_rule():
    state = PipelineState(config=PipelineConfig(
        wall_region_room_support_min_rooms=2,
        wall_region_room_support_radius_px=6,
        wall_region_room_support_min_overlap=0.20,
    ))
    state.image = np.full((300, 500), 255, np.uint8)
    state.rooms = [
        Room("R1", [Point(30, 30), Point(220, 30), Point(220, 270), Point(30, 270)],
             area=45600),
        Room("R2", [Point(240, 30), Point(470, 30), Point(470, 270), Point(240, 270)],
             area=55200),
    ]
    # Thin structural partition follows both room boundaries. The parallel
    # floating rule crosses room interiors and has no free-space boundary role.
    state.walls = [
        wall("PARTITION", 230, 30, 230, 270, 6),
        wall("FLOATING", 70, 150, 190, 150, 6),
    ]

    semantic_masks.run(state)

    assert state.wall_polygon_mask[150, 230] == 255
    assert state.wall_polygon_mask[150, 130] == 0
    assert state.rejected_wall_candidate_mask[150, 130] == 255
    assert state.debug.segment_counts["13_supported_walls"] == 1
    assert state.debug.segment_counts["13_rejected_wall_candidates"] == 1


def test_exterior_ring_uses_room_inner_face_and_supported_shell_thickness():
    state = PipelineState(config=PipelineConfig(
        wall_thickness_max_px=50,
        wall_region_axis_min_run_px=21,
        exterior_wall_min_side_support=0.45,
    ))
    state.image = np.full((320, 440), 255, np.uint8)
    state.binary_cleaned = np.zeros(state.image.shape, np.uint8)
    state.structural_core_mask = np.zeros(state.image.shape, np.uint8)
    state.structural_core_mask[40:280, 60:380] = 255
    state.rooms = [Room(
        "R1", [Point(100, 80), Point(340, 80), Point(340, 240), Point(100, 240)],
        label="ROOM", area=38400,
    )]
    # Inner room faces are at x=100/340 and y=80/240. Sustained outer faces
    # 30 px away provide a consistent, independently observed exterior shell.
    cv2.line(state.binary_cleaned, (70, 50), (70, 270), 255, 2)
    cv2.line(state.binary_cleaned, (370, 50), (370, 270), 255, 2)
    cv2.line(state.binary_cleaned, (70, 50), (370, 50), 255, 2)
    cv2.line(state.binary_cleaned, (70, 270), (370, 270), 255, 2)

    semantic_masks.run(state)

    assert state.wall_repaired_mask[60, 220] == 255
    assert state.wall_repaired_mask[100, 85] == 255
    assert state.wall_repaired_mask[160, 220] == 0


def test_exterior_rectangle_is_not_inferred_for_nonrectangular_core():
    state = PipelineState(config=PipelineConfig(
        wall_thickness_max_px=50,
        exterior_wall_min_side_support=0.45,
        exterior_wall_min_rectangularity=0.85,
    ))
    state.image = np.full((320, 440), 255, np.uint8)
    state.binary_cleaned = np.zeros(state.image.shape, np.uint8)
    state.structural_core_mask = np.zeros(state.image.shape, np.uint8)
    state.structural_core_mask[40:280, 60:180] = 255
    state.structural_core_mask[160:280, 180:380] = 255
    state.rooms = [Room(
        "R1", [Point(100, 80), Point(340, 80), Point(340, 240), Point(100, 240)],
        label="ROOM", area=38400,
    )]
    cv2.rectangle(state.binary_cleaned, (70, 50), (370, 270), 255, 2)

    semantic_masks.run(state)

    assert not np.any(state.wall_repaired_mask)
