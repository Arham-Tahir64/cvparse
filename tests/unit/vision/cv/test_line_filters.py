"""Tests for module 04 - line filtering, against the spec's test criteria."""
import numpy as np

from vision.cv import line_filters
from vision.cv.config import PipelineConfig
from vision.cv.models import LineSegment, PipelineState, Point, TextElement


def seg(x1, y1, x2, y2, thickness=1.0, sid=""):
    return LineSegment(Point(x1, y1), Point(x2, y2), thickness=thickness, id=sid)


def make_state(lines, texts=(), binary=None, roi_mask=None, core_mask=None):
    state = PipelineState(config=PipelineConfig())
    state.raw_lines = list(lines)
    state.raw_texts = list(texts)
    state.binary = binary if binary is not None else np.zeros((500, 500), np.uint8)
    state.image = np.full((500, 500), 255, np.uint8)
    if roi_mask is None:
        roi_mask = np.full((500, 500), 255, np.uint8)
    state.structural_roi_mask = roi_mask
    state.structural_core_mask = core_mask
    return state


def classify(lines, **kwargs):
    state = make_state(lines, **kwargs)
    line_filters.run(state)
    return state.classified_lines


def test_outside_roi_is_text_baseline():
    mask = np.zeros((500, 500), np.uint8)
    mask[:250, :] = 255
    out = classify([seg(100, 400, 200, 400)], roi_mask=mask)
    assert out[0].classification == "text_baseline"


def test_intersecting_text_bbox_is_text_baseline():
    text = TextElement("KITCHEN", (90, 95, 210, 115), 0.9)
    out = classify([seg(80, 100, 220, 100)], texts=[text])
    assert out[0].classification == "text_baseline"


def test_short_segment_near_text_is_leader():
    text = TextElement("NOTE", (200, 200, 260, 220), 0.9)
    out = classify([seg(150, 250, 195, 225)], texts=[text])
    assert out[0].classification == "leader"


def test_thin_long_with_ticks_is_dimension():
    main = seg(100, 100, 250, 100, thickness=1.0)
    tick1 = seg(100, 95, 100, 105, thickness=1.0)
    tick2 = seg(250, 95, 250, 105, thickness=1.0)
    out = classify([main, tick1, tick2])
    assert out[0].classification == "dimension"


def test_thin_with_imperial_text_is_dimension():
    text = TextElement("12'-6\"", (160, 103, 200, 112), 0.9)
    # keep the text far enough to not trigger rule 1/2 but within 20 px of midpoint
    out = classify([seg(100, 120, 250, 120, thickness=1.0)], texts=[text])
    assert out[0].classification == "dimension"


def test_thin_with_metric_text_is_dimension():
    text = TextElement("1500mm", (160, 103, 200, 112), 0.9)
    out = classify([seg(100, 120, 250, 120, thickness=1.0)], texts=[text])
    assert out[0].classification == "dimension"


def test_thick_with_ticks_not_dimension():
    main = seg(100, 100, 250, 100, thickness=5.0)
    tick1 = seg(100, 95, 100, 105)
    tick2 = seg(250, 95, 250, 105)
    out = classify([main, tick1, tick2])
    assert out[0].classification != "dimension"


def test_hatch_cluster():
    lines = [seg(100 + i * 4, 100, 120 + i * 4, 120, sid=f"h{i}") for i in range(10)]
    out = classify(lines)
    assert all(s.classification == "hatch" for s in out)


def test_lone_short_segment_unknown():
    out = classify([seg(100, 100, 120, 120)])
    assert out[0].classification == "unknown"


def test_dashed_long_line_is_grid():
    binary = np.zeros((500, 500), np.uint8)
    for x0 in range(50, 450, 20):
        binary[300, x0:x0 + 12] = 255  # 12 ink, 8 gap
    out = classify([seg(50, 300, 450, 300, thickness=1.0)], binary=binary)
    # thin + long: ensure grid beats dimension (no ticks/text present)
    assert out[0].classification == "grid"


def test_thick_long_unclustered_is_unknown():
    binary = np.zeros((500, 500), np.uint8)
    binary[300, 100:300] = 255  # solid line, no dashes
    out = classify([seg(100, 300, 300, 300, thickness=6.0)], binary=binary)
    assert out[0].classification == "unknown"


def test_thin_line_outside_core_mask_is_dimension():
    core = np.zeros((500, 500), np.uint8)
    core[100:400, 100:400] = 255  # plan body
    # thin margin line above the plan (dimension string territory)
    out = classify([seg(120, 50, 380, 50, thickness=1.5)], core_mask=core)
    assert out[0].classification == "dimension"


def test_thick_line_outside_core_mask_not_dimension():
    core = np.zeros((500, 500), np.uint8)
    core[100:400, 100:400] = 255
    out = classify([seg(120, 50, 380, 50, thickness=6.0)], core_mask=core)
    assert out[0].classification != "dimension"


def test_thin_line_inside_core_mask_stays_unknown():
    core = np.zeros((500, 500), np.uint8)
    core[100:400, 100:400] = 255
    out = classify([seg(150, 250, 350, 250, thickness=1.5)], core_mask=core)
    assert out[0].classification == "unknown"


def test_diagonal_hatch_cluster_long_lines():
    # 10 long parallel 45-degree lines, 12 px apart: dropped-ceiling hatch
    lines = [seg(100 + i * 12, 100, 250 + i * 12, 250, sid=f"d{i}")
             for i in range(10)]
    out = classify(lines)
    assert all(s.classification == "hatch" for s in out)


def test_lone_long_diagonal_stays_unknown():
    out = classify([seg(100, 100, 300, 280)])
    assert out[0].classification == "unknown"


def test_long_axis_aligned_lone_line_stays_unknown():
    out = classify([seg(50, 250, 450, 250, thickness=2.0)])
    assert out[0].classification == "unknown"


def test_slash_ticks_detected_as_dimension():
    # 45-degree slash ticks at both endpoints (common arch style)
    main = seg(100, 100, 250, 100, thickness=1.0)
    tick1 = seg(95, 105, 105, 95, thickness=1.0)
    tick2 = seg(245, 105, 255, 95, thickness=1.0)
    out = classify([main, tick1, tick2])
    assert out[0].classification == "dimension"


def test_output_length_matches_input():
    lines = [seg(10 * i, 5, 10 * i + 30, 5, sid=str(i)) for i in range(12)]
    out = classify(lines)
    assert len(out) == len(lines)
