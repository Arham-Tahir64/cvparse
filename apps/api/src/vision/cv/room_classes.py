"""Stable semantic room classes and annotation palette."""
from __future__ import annotations


# RGB colour and opacity. These classes are architectural semantics, not
# drawing-specific coordinates, so the palette remains stable across plans.
ROOM_CLASS_STYLES: dict[str, tuple[tuple[int, int, int], float]] = {
    "guest_suite": ((255, 222, 91), 0.42),
    "bath_4": ((91, 222, 232), 0.35),
    "gym_yoga": ((164, 119, 238), 0.38),
    "laundry": ((100, 211, 180), 0.35),
    "linen": ((238, 166, 83), 0.38),
    "mechanical": ((255, 88, 111), 0.35),
    "storage": ((246, 145, 60), 0.38),
    "stair_circulation": ((190, 190, 190), 0.40),
    "recreation_room": ((135, 207, 104), 0.38),
    "room": ((173, 199, 232), 0.25),
}

ROOM_CLASS_NAMES = {
    "guest_suite": "Guest Suite",
    "bath_4": "Bath 4",
    "gym_yoga": "Gym/Yoga",
    "laundry": "Laundry",
    "linen": "Linen",
    "mechanical": "Mechanical",
    "storage": "Storage",
    "stair_circulation": "Stair/Circulation",
    "recreation_room": "Recreation Room",
    "room": "Room",
}


def room_class_key(label: str | None) -> str:
    value = " ".join((label or "").upper().replace(".", "").split())
    if "GUEST" in value:
        return "guest_suite"
    if "BATH" in value:
        return "bath_4"
    if "GYM" in value or "YOGA" in value:
        return "gym_yoga"
    if "LAUNDRY" in value or "LNDRY" in value:
        return "laundry"
    if "LINEN" in value:
        return "linen"
    if "MECH" in value:
        return "mechanical"
    if "STORAGE" in value:
        return "storage"
    if "STAIR" in value or "CIRCULATION" in value or value == "UP":
        return "stair_circulation"
    if "REC" in value or "RECREATION" in value:
        return "recreation_room"
    return "room"
