"""Module 10 - OCR Room Labeling.

High-confidence OCR pass on the original image, vocabulary matching, and
label assignment to rooms by containment then proximity.
"""
from __future__ import annotations

import logging
import os
import re

import cv2
import numpy as np
from shapely.geometry import Point as ShapelyPoint, Polygon

from . import ocr_engines
from .line_detection import run_ocr_first_pass  # re-exported per spec
from .models import PipelineError, PipelineState, Room, TextElement

logger = logging.getLogger("flowbuildr.cv.ocr_labeling")

MODULE = "10_ocr_labeling"

__all__ = ["run", "run_ocr_first_pass"]

_NORMALIZE_RE = re.compile(r"[^A-Z0-9 ]+")


def run(state: PipelineState) -> PipelineState:
    config = state.config

    engine = ocr_engines.get_engine(config.ocr_engine)
    if engine is None:
        raise PipelineError(
            MODULE,
            "no OCR engine available: install paddleocr "
            "(pip install paddleocr) or pytesseract plus the tesseract binary",
        )

    texts = None
    future = state.ocr_second_pass_future
    if future is not None:
        state.ocr_second_pass_future = None
        try:
            texts = [
                t for t in future.result()
                if t.confidence >= config.ocr_second_pass_confidence
            ]
        except Exception as exc:
            logger.warning("background second-pass OCR failed, rereading: %s", exc)
    if texts is None:
        texts = engine.read(state.image, config.ocr_second_pass_confidence)
    logger.debug("second-pass OCR found %d text elements", len(texts))

    matched = [(t, _match_vocab(t.text, config.room_label_vocab)) for t in texts]
    matched = [(t, m) for t, m in matched if m is not None]

    polygons = {
        room.id: Polygon([(p.x, p.y) for p in room.polygon]) for room in state.rooms
    }
    room_by_id = {room.id: room for room in state.rooms}
    assigned_texts: dict[str, list[TextElement]] = {rid: [] for rid in polygons}

    # Phase 1 - containment
    unplaced = []
    for text, _vocab in matched:
        center = ShapelyPoint(text.center.x, text.center.y)
        holder = next((rid for rid, poly in polygons.items() if poly.contains(center)), None)
        if holder is None:
            unplaced.append((text, _vocab))
            continue
        _assign(room_by_id[holder], text)
        assigned_texts[holder].append(text)

    # Phase 2 - proximity
    for text, _vocab in unplaced:
        center = ShapelyPoint(text.center.x, text.center.y)
        best_rid, best_dist = None, config.label_room_max_distance_px
        for rid, poly in polygons.items():
            d = poly.distance(center)
            if d <= best_dist:
                best_rid, best_dist = rid, d
        if best_rid is not None:
            _assign(room_by_id[best_rid], text)
            assigned_texts[best_rid].append(text)

    # Multi-word combination
    for rid, texts_in_room in assigned_texts.items():
        if len(texts_in_room) < 2:
            continue
        ordered = sorted(texts_in_room, key=lambda t: (t.bbox[1], t.bbox[0]))
        combined = " ".join(t.text.strip() for t in ordered)
        # only replace when the combined string as a whole is a vocab entry
        if _normalize(combined) in config.room_label_vocab:
            room = room_by_id[rid]
            room.label = combined
            room.label_confidence = min(t.confidence for t in ordered)

    labeled = sum(1 for r in state.rooms if r.label)
    logger.info("labeled %d/%d rooms", labeled, len(state.rooms))
    for room in state.rooms:
        if not room.label:
            logger.debug("room %s has no label", room.id)

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "10_labels.png"),
            visualize(state, state.image),
        )
    return state


def _assign(room: Room, text: TextElement) -> None:
    if text.confidence > room.label_confidence or room.label is None:
        room.label = text.text.strip()
        room.label_confidence = text.confidence


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.upper()).strip()


def _match_vocab(text: str, vocab):
    """Longest vocabulary entry appearing as a substring of the normalized text."""
    normalized = _normalize(text)
    if not normalized:
        return None
    best = None
    for entry in vocab:
        if entry in normalized and (best is None or len(entry) > len(best)):
            best = entry
    return best


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for room in state.rooms:
        if not room.label or not room.polygon:
            continue
        cx = int(sum(p.x for p in room.polygon) / len(room.polygon))
        cy = int(sum(p.y for p in room.polygon) / len(room.polygon))
        cv2.putText(overlay, room.label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return overlay
