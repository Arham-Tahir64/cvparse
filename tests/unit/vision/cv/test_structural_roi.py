"""Tests for module 02 - structural ROI, against the spec's test criteria."""
import cv2
import numpy as np

from vision.cv import structural_roi
from vision.cv.config import PipelineConfig
from vision.cv.models import PipelineState


def make_state(binary: np.ndarray, **config_kwargs) -> PipelineState:
    state = PipelineState(config=PipelineConfig(**config_kwargs))
    state.binary = binary
    state.image = np.where(binary > 0, 0, 255).astype(np.uint8)
    return state


def draw_plan(binary, x0, y0, x1, y1, thickness=6):
    cv2.rectangle(binary, (x0, y0), (x1, y1), 255, thickness=thickness)


def test_plan_covered_title_block_excluded():
    binary = np.zeros((800, 1000), np.uint8)
    draw_plan(binary, 100, 100, 600, 600)
    # corner title block: dense small text-like blobs
    for y in range(650, 780, 12):
        for x in range(750, 980, 15):
            binary[y:y + 4, x:x + 8] = 255
    state = make_state(binary)
    structural_roi.run(state)
    assert state.structural_roi_mask[350, 350] == 255           # inside plan
    assert state.structural_roi_mask[720, 900] == 0             # title block
    assert state.binary_masked[720:780, 750:980].sum() == 0


def test_plan_touching_right_edge_included():
    binary = np.zeros((600, 800), np.uint8)
    draw_plan(binary, 300, 100, 795, 500)
    state = make_state(binary)
    structural_roi.run(state)
    assert state.binary_masked[100:110, 780:796].sum() > 0


def test_blank_image_falls_back_full_roi():
    binary = np.zeros((400, 400), np.uint8)
    state = make_state(binary)
    structural_roi.run(state)
    assert (state.structural_roi_mask == 255).all()
    assert any("fall" in m.lower() for m in state.debug.messages)


def test_binary_unmodified():
    binary = np.zeros((400, 400), np.uint8)
    draw_plan(binary, 50, 50, 350, 350)
    original = binary.copy()
    state = make_state(binary)
    structural_roi.run(state)
    assert (state.binary == original).all()


def test_masked_zero_outside_equal_inside():
    binary = np.zeros((600, 800), np.uint8)
    draw_plan(binary, 100, 100, 500, 500)
    binary[550:560, 700:790] = 255  # far-away blob
    state = make_state(binary)
    structural_roi.run(state)
    mask = state.structural_roi_mask
    assert (state.binary_masked[mask == 0] == 0).all()
    inside = mask == 255
    assert (state.binary_masked[inside] == state.binary[inside]).all()


def test_mask_binary_values_and_dims():
    binary = np.zeros((300, 500), np.uint8)
    draw_plan(binary, 50, 50, 450, 250)
    state = make_state(binary)
    structural_roi.run(state)
    assert state.structural_roi_mask.shape == binary.shape
    assert set(np.unique(state.structural_roi_mask)) <= {0, 255}


def test_core_mask_populated_and_tighter_than_roi():
    binary = np.zeros((600, 800), np.uint8)
    draw_plan(binary, 100, 100, 500, 500)
    state = make_state(binary)
    structural_roi.run(state)
    core = state.structural_core_mask
    assert core is not None
    assert core.shape == binary.shape
    # core (pre-dilation) covers the plan but is strictly smaller than the ROI
    assert core[300, 300] == 255
    assert (core > 0).sum() < (state.structural_roi_mask > 0).sum()
    # ROI-but-not-core band exists around the plan
    assert ((state.structural_roi_mask > 0) & (core == 0)).any()


def test_core_mask_full_on_fallback():
    binary = np.zeros((400, 400), np.uint8)
    state = make_state(binary)
    structural_roi.run(state)
    assert (state.structural_core_mask == 255).all()


def test_mid_sheet_schedule_table_plan_still_captured():
    binary = np.zeros((800, 1200), np.uint8)
    draw_plan(binary, 80, 80, 700, 700)
    # interior walls make the plan seed denser
    cv2.line(binary, (80, 400), (700, 400), 255, 6)
    cv2.line(binary, (400, 80), (400, 700), 255, 6)
    # schedule table mid-right: short line rows
    for y in range(300, 500, 18):
        cv2.line(binary, (900, y), (1150, y), 255, 2)
    state = make_state(binary)
    structural_roi.run(state)
    assert state.structural_roi_mask[400, 400] == 255
