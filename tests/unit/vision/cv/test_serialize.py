"""Tests for module 11 - serialization and annotation adapter."""
import json
import math

import numpy as np

from vision.adapters.annotation_adapter import to_annotation_document
from vision.cv import serialize
from vision.cv.models import (
    CVTakeoffResult, DebugInfo, Door, Gap, LineSegment, PlanMetadata, Point,
    Room, Wall,
)


def make_wall(wid="W0001", confidence=0.9, start=(10, 20), end=(200, 20)):
    cl = LineSegment(Point(*start), Point(*end), thickness=8.0)
    return Wall(
        id=wid, orientation="H", centerline=cl, thickness=8.0,
        visual_thickness=9.5, merge_kind="paired_faces", fit_support_ratio=0.87,
        merge_confidence=confidence, source_ids=["L00001", "L00002"],
        length_px=cl.length,
    )


def make_result(walls=(), doors=(), rooms=(), gaps=()):
    return CVTakeoffResult(
        walls=list(walls), gaps=list(gaps), doors=list(doors), windows=[],
        rooms=list(rooms),
        metadata=PlanMetadata(None, 800, 600, 200, 0, len(walls), len(rooms)),
        debug=DebugInfo(),
    )


def full_result():
    door = Door(
        id="D0001", position=Point(100, 20), swing_end=Point(140, 60),
        radius=40.0, wall_id="W0001", swing_direction="cw",
        swing_arc=[Point(140, 20), Point(128, 48), Point(100, 60)],
        confidence=0.81,
    )
    room = Room(id="R0001", polygon=[Point(0, 0), Point(100, 0), Point(100, 80),
                                     Point(0, 80)], label="KITCHEN",
                label_confidence=0.92, area=8000.0)
    gap = Gap(id="G0001", wall_id="W0001", orientation="H", center=Point(100, 20),
              width_px=40.0, bbox=(80, 0, 120, 40), kind="door",
              wall_break_score=0.25, opening_fill_ratio=0.95)
    return make_result([make_wall(), make_wall("W0002", confidence=0.4)],
                       [door], [room], [gap])


def test_round_trip_json():
    data = serialize.to_json_dict(full_result())
    text = json.dumps(data)
    parsed = json.loads(text)
    assert parsed["walls"][0]["id"] == "W0001"
    assert parsed["walls"][0]["merge_kind"] == "paired_faces"
    assert parsed["walls"][0]["source_ids"] == ["L00001", "L00002"]
    assert parsed["doors"][0]["swing_direction"] == "cw"
    assert len(parsed["doors"][0]["swing_arc"]) == 3
    assert parsed["doors"][0]["confidence"] == 0.81
    assert parsed["rooms"][0]["label"] == "KITCHEN"
    assert parsed["metadata"]["wall_count"] == 2


def test_schema_version():
    data = serialize.to_json_dict(full_result())
    assert data["schema_version"] == "1.0.0"


def test_empty_result_valid():
    data = serialize.to_json_dict(make_result())
    json.dumps(data)
    assert data["walls"] == []
    assert data["rooms"] == []


def test_nan_serialized_as_null():
    wall = make_wall(start=(float("nan"), 20))
    data = serialize.to_json_dict(make_result([wall]))
    assert data["walls"][0]["start"]["x"] is None
    assert data["walls"][0]["start"]["y"] == 20


def test_rounding_three_decimals():
    wall = make_wall(start=(10.123456, 20.98765))
    data = serialize.to_json_dict(make_result([wall]))
    assert data["walls"][0]["start"]["x"] == 10.123
    assert data["walls"][0]["start"]["y"] == 20.988


def test_low_confidence_wall_orange_in_svg():
    svg = serialize.to_svg(full_result())
    assert "#ff7f0e" in svg
    assert "#d62728" in svg


def test_svg_uses_visual_thickness():
    svg = serialize.to_svg(make_result([make_wall()]))
    assert 'stroke-width="9.5"' in svg


def test_svg_roi_boundary_purple_dashed():
    mask = np.zeros((600, 800), np.uint8)
    mask[100:500, 100:700] = 255
    svg = serialize.to_svg(full_result(), roi_mask=mask)
    assert "#9467bd" in svg
    assert "stroke-dasharray" in svg


def test_gap_serializes_with_kind():
    data = serialize.to_json_dict(full_result())
    assert data["gaps"][0]["kind"] == "door"
    assert data["gaps"][0]["wall_break_score"] == 0.25


def test_adapter_exports_door_geometry():
    doc = to_annotation_document(full_result())
    door = next(element for element in doc["elements"] if element["type"] == "door")
    assert door["geometry"]["kind"] == "swing"
    assert len(door["geometry"]["arc"]) == 3
    assert door["relations"]["wall_id"] == "W0001"


def test_adapter_wall_element():
    doc = to_annotation_document(make_result([make_wall()]))
    wall_el = [e for e in doc["elements"] if e["type"] == "wall"][0]
    assert wall_el["id"] == "W0001"
    assert wall_el["geometry"]["kind"] == "segment"
    assert wall_el["geometry"]["x1"] == 10
    assert wall_el["geometry"]["thickness_px"] == 9.5
    assert wall_el["review_state"] == "pending"


def test_adapter_relations():
    doc = to_annotation_document(make_result([make_wall()]))
    rel = doc["elements"][0]["relations"]
    assert rel["fit_support_ratio"] == 0.87
    assert rel["merge_confidence"] == 0.9
    assert rel["merge_kind"] == "paired_faces"


def test_adapter_room_label():
    doc = to_annotation_document(full_result())
    room_el = [e for e in doc["elements"] if e["type"] == "room"][0]
    assert room_el["label"] == "KITCHEN"
    assert room_el["area_px"] == 8000.0


def test_adapter_empty_document():
    doc = to_annotation_document(make_result())
    assert doc == {"elements": []}
