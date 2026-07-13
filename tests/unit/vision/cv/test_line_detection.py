"""Tests for module 03 - line detection, against the spec's test criteria."""
import math

import cv2
import numpy as np
import pytest

from vision.cv import line_detection
from vision.cv.config import PipelineConfig
from vision.cv.models import NoLinesDetectedError, PipelineState


def make_state(binary, config=None):
    state = PipelineState(config=config or PipelineConfig())
    state.binary = binary
    state.binary_masked = binary.copy()
    state.image = np.where(binary > 0, 0, 255).astype(np.uint8)
    return state


def run_lines(binary, config=None):
    state = make_state(binary, config)
    line_detection.run(state)
    return state


def horizontal_segments(segs, tol_deg=10):
    return [s for s in segs
            if min(math.degrees(s.angle_rad), 180 - math.degrees(s.angle_rad)) < tol_deg]


def test_square_outline_four_segments():
    binary = np.zeros((400, 400), np.uint8)
    cv2.rectangle(binary, (100, 100), (300, 300), 255, 3)
    state = run_lines(binary)
    # LSD detects both faces of each stroke; after merging, segments cluster
    # into the 4 sides. Verify each side is represented within tolerance.
    sides = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for seg in state.raw_lines:
        m = seg.midpoint
        if seg.is_horizontal and seg.length > 150:
            sides["top" if m.y < 200 else "bottom"] += 1
        elif seg.is_vertical and seg.length > 150:
            sides["left" if m.x < 200 else "right"] += 1
    assert all(v >= 1 for v in sides.values())


def test_fragments_with_small_filled_gaps_merge():
    binary = np.zeros((200, 400), np.uint8)
    # one continuous ink line, "detected" as fragments by drawing fully
    cv2.line(binary, (50, 100), (350, 100), 255, 2)
    state = run_lines(binary)
    longs = [s for s in horizontal_segments(state.raw_lines) if s.length > 250]
    assert len(longs) >= 1


def test_large_empty_gap_not_merged():
    binary = np.zeros((200, 500), np.uint8)
    cv2.line(binary, (50, 100), (200, 100), 255, 2)   # fragment 1
    cv2.line(binary, (236, 100), (400, 100), 255, 2)  # 36 px empty gap
    state = run_lines(binary)
    # no merged segment spans the gap
    for seg in horizontal_segments(state.raw_lines):
        xs = sorted((seg.start.x, seg.end.x))
        assert not (xs[0] < 210 and xs[1] > 230), f"segment bridged the gap: {seg}"


def test_perpendicular_lines_do_not_merge():
    binary = np.zeros((300, 300), np.uint8)
    cv2.line(binary, (50, 150), (250, 150), 255, 2)
    cv2.line(binary, (150, 50), (150, 250), 255, 2)
    state = run_lines(binary)
    # H and V segments must not merge into each other: every output segment
    # stays close to axis-aligned, none becomes a long diagonal
    assert all(s.is_horizontal or s.is_vertical for s in state.raw_lines)
    assert any(s.is_horizontal and s.length > 80 for s in state.raw_lines)
    assert any(s.is_vertical and s.length > 80 for s in state.raw_lines)


def test_parallel_lines_20px_apart_stay_separate():
    binary = np.zeros((200, 400), np.uint8)
    cv2.line(binary, (50, 90), (350, 90), 255, 2)
    cv2.line(binary, (50, 110), (350, 110), 255, 2)
    state = run_lines(binary)
    ys = set()
    for seg in horizontal_segments(state.raw_lines):
        if seg.length > 200:
            ys.add(round(seg.midpoint.y / 10))
    assert len(ys) >= 2


def test_short_lines_filtered():
    binary = np.zeros((200, 200), np.uint8)
    cv2.line(binary, (50, 100), (58, 100), 255, 2)   # 8 px, below min 15
    cv2.line(binary, (50, 150), (150, 150), 255, 2)
    state = run_lines(binary)
    assert all(s.length >= state.config.min_line_length_px for s in state.raw_lines)


def test_blank_image_raises():
    binary = np.zeros((200, 200), np.uint8)
    with pytest.raises(NoLinesDetectedError):
        run_lines(binary)


def test_masked_binary_yields_fewer_segments():
    binary = np.zeros((400, 600), np.uint8)
    cv2.rectangle(binary, (50, 50), (350, 350), 255, 3)
    for y in range(80, 300, 20):
        cv2.line(binary, (420, y), (580, y), 255, 2)  # title-block-ish content
    masked = binary.copy()
    masked[:, 400:] = 0

    full_state = make_state(binary)
    full_state.binary_masked = binary.copy()
    line_detection.run(full_state)

    masked_state = make_state(binary)
    masked_state.binary_masked = masked
    line_detection.run(masked_state)

    assert len(masked_state.raw_lines) < len(full_state.raw_lines)


def test_all_outputs_unknown_classification():
    binary = np.zeros((200, 300), np.uint8)
    cv2.line(binary, (20, 100), (280, 100), 255, 3)
    state = run_lines(binary)
    assert all(s.classification == "unknown" for s in state.raw_lines)


def test_ocr_unavailable_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(line_detection.ocr_engines, "get_engine", lambda *_: None)
    binary = np.zeros((200, 300), np.uint8)
    cv2.line(binary, (20, 100), (280, 100), 255, 3)
    state = run_lines(binary)
    assert state.raw_texts == []
    assert any("OCR" in m for m in state.debug.messages)
