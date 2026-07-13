"""Tests for module 10 - OCR room labeling, against the spec's test criteria.

Uses a fake OCR engine; no paddle/tesseract needed.
"""
import numpy as np
import pytest

from vision.cv import ocr_labeling
from vision.cv.config import PipelineConfig
from vision.cv.models import PipelineError, PipelineState, Point, Room, TextElement


class FakeEngine:
    name = "fake"

    def __init__(self, elements):
        self.elements = elements

    def read(self, image, confidence_threshold):
        return [e for e in self.elements if e.confidence >= confidence_threshold]


def text(s, x, y, conf=0.9, w=60, h=16):
    return TextElement(s, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), conf)


def room(rid, x0, y0, x1, y1):
    return Room(id=rid, polygon=[
        Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)
    ], area=(x1 - x0) * (y1 - y0))


def run_with(monkeypatch, rooms, elements):
    state = PipelineState(config=PipelineConfig())
    state.image = np.full((600, 800), 255, np.uint8)
    state.rooms = list(rooms)
    monkeypatch.setattr(
        ocr_labeling.ocr_engines, "get_engine", lambda *_: FakeEngine(elements)
    )
    ocr_labeling.run(state)
    return state


def test_text_inside_room(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("KITCHEN", 250, 250)])
    assert state.rooms[0].label == "KITCHEN"


def test_text_near_room_assigned(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("BATH", 430, 250)])  # 30 px outside
    assert state.rooms[0].label == "BATH"


def test_text_far_from_rooms_unassigned(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("BATH", 700, 550)])  # ~200+ px away
    assert state.rooms[0].label is None


def test_original_text_preserved(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("BED 2", 250, 250)])
    assert state.rooms[0].label == "BED 2"


def test_longest_vocab_match():
    assert ocr_labeling._match_vocab(
        "BATHROOM", PipelineConfig().room_label_vocab) == "BATHROOM"


def test_dimension_text_no_match():
    assert ocr_labeling._match_vocab(
        "12'-6\"", PipelineConfig().room_label_vocab) is None


def test_higher_confidence_wins(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("DEN", 200, 200, conf=0.61),
                      text("OFFICE", 300, 300, conf=0.9)])
    assert state.rooms[0].label == "OFFICE"


def test_multi_word_combination(monkeypatch):
    state = run_with(monkeypatch, [room("R0001", 100, 100, 400, 400)],
                     [text("MASTER", 250, 230), text("BEDROOM", 250, 260)])
    assert state.rooms[0].label == "MASTER BEDROOM"


def test_room_with_no_text_unlabeled(monkeypatch):
    state = run_with(monkeypatch,
                     [room("R0001", 100, 100, 400, 400),
                      room("R0002", 450, 100, 700, 400)],
                     [text("KITCHEN", 250, 250)])
    assert state.rooms[1].label is None


def test_no_engine_raises(monkeypatch):
    state = PipelineState(config=PipelineConfig())
    state.image = np.full((100, 100), 255, np.uint8)
    monkeypatch.setattr(ocr_labeling.ocr_engines, "get_engine", lambda *_: None)
    with pytest.raises(PipelineError):
        ocr_labeling.run(state)
