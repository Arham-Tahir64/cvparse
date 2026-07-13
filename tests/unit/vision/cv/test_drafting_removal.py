"""Tests for the protected drafting-removal pass."""
import cv2
import numpy as np

from vision.cv import drafting_removal
from vision.cv.config import PipelineConfig
from vision.cv.models import LineSegment, PipelineState, Point, TextElement


def segment(sid, x1, y1, x2, y2, classification):
    return LineSegment(
        Point(x1, y1), Point(x2, y2), thickness=1.5,
        id=sid, classification=classification,
    )


def make_state():
    config = PipelineConfig(drafting_repair_gap_px=9)
    state = PipelineState(config=config)
    state.image = np.full((300, 500), 255, np.uint8)
    state.binary = np.zeros((300, 500), np.uint8)
    state.binary_masked = state.binary.copy()
    state.semantic_plan_mask = np.full_like(state.binary, 255)
    return state


def test_dimension_removed_while_paired_wall_faces_survive():
    state = make_state()
    cv2.line(state.binary_masked, (50, 100), (450, 100), 255, 2)
    cv2.line(state.binary_masked, (50, 112), (450, 112), 255, 2)
    cv2.line(state.binary_masked, (50, 220), (450, 220), 255, 1)
    state.binary = state.binary_masked.copy()
    state.classified_lines = [
        segment("W1", 50, 100, 450, 100, "unknown"),
        segment("W2", 50, 112, 450, 112, "unknown"),
        segment("D1", 50, 220, 450, 220, "dimension"),
    ]
    state.raw_texts = [TextElement("12'-6\"", (230, 205, 275, 218), 0.95)]

    drafting_removal.run(state)

    assert state.drafting_mask[220, 250] == 255
    assert state.interior_drafting_mask[220, 250] == 255
    assert state.binary_cleaned[220, 250] == 0
    assert state.binary_cleaned[100, 250] == 255
    assert state.binary_cleaned[112, 250] == 255
    assert state.debug.segment_counts["05_protected_walls"] == 1


def test_curved_door_geometry_is_not_removed():
    state = make_state()
    cv2.ellipse(state.binary_masked, (250, 150), (60, 60), 0, 0, 90, 255, 2)
    cv2.line(state.binary_masked, (50, 250), (450, 250), 255, 1)
    state.binary = state.binary_masked.copy()
    state.classified_lines = [
        segment("D1", 50, 250, 450, 250, "dimension"),
    ]

    drafting_removal.run(state)

    assert state.binary_cleaned[150, 310] > 0
    assert state.binary_cleaned[210, 250] > 0
    assert state.binary_cleaned[250, 250] == 0


def test_ink_outside_semantic_plan_envelope_is_removed():
    state = make_state()
    state.semantic_plan_mask[:] = 0
    state.semantic_plan_mask[50:250, 100:400] = 255
    cv2.line(state.binary_masked, (20, 20), (480, 20), 255, 2)
    cv2.line(state.binary_masked, (120, 80), (380, 80), 255, 2)
    state.binary = state.binary_masked.copy()

    drafting_removal.run(state)

    assert state.binary_cleaned[20, 250] == 0
    assert state.binary_cleaned[80, 250] == 255
    assert state.drafting_mask[20, 250] == 255
    assert state.interior_drafting_mask[20, 250] == 0
