"""Tests for stable semantic room class normalization."""

from vision.cv.room_classes import ROOM_CLASS_STYLES, room_class_key


def test_room_labels_map_to_stable_architectural_classes():
    assert room_class_key("Guest Suite") == "guest_suite"
    assert room_class_key("BATH 4") == "bath_4"
    assert room_class_key("GYM/YOGA") == "gym_yoga"
    assert room_class_key("MECH.") == "mechanical"
    assert room_class_key("UP") == "stair_circulation"
    assert room_class_key("REC ROOM AREA") == "recreation_room"


def test_all_room_classes_have_distinct_rgb_colours():
    colours = [style[0] for key, style in ROOM_CLASS_STYLES.items() if key != "room"]
    assert len(colours) == len(set(colours))
