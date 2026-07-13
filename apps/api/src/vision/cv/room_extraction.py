"""Module 09 - Room Extraction.

Primary path: treat the wall network as a planar graph and extract bounded
faces. Fallback: flood-fill segmentation of the rasterized wall mask.
"""
from __future__ import annotations

import logging
import os

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import Polygon
from skimage import measure

from .models import IdGenerator, NoRoomsExtractedError, PipelineState, Point, Room

logger = logging.getLogger("flowbuildr.cv.room_extraction")

MODULE = "09_room_extraction"


def run(state: PipelineState) -> PipelineState:
    config = state.config
    image_area = float(state.image.shape[0] * state.image.shape[1])
    id_gen = IdGenerator("R")

    rooms = _semantic_raster_rooms(state, image_area, id_gen)
    if len(rooms) >= config.semantic_room_min_seeds:
        state.debug.messages.append(
            f"semantic raster path selected with {len(rooms)} seeded rooms"
        )
        planar_ok = True
    else:
        rooms = []
        planar_ok = False

    if not rooms:
        try:
            rooms, planar_ok = _planar_graph_rooms(state, image_area, id_gen)
        except Exception:
            logger.exception("planar graph path failed; falling back to flood fill")

    if (not planar_ok or not rooms) and config.enable_floodfill_fallback:
        logger.info("using flood-fill fallback (planar_ok=%s, rooms=%d)",
                    planar_ok, len(rooms))
        rooms = _floodfill_rooms(state, image_area, IdGenerator("R"))

    if not rooms:
        raise NoRoomsExtractedError(MODULE, "no rooms found by either extraction path")

    state.rooms = rooms
    state.debug.segment_counts["09_rooms"] = len(rooms)
    logger.info("extracted %d rooms", len(rooms))

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "09_rooms.png"),
            visualize(state, state.image),
        )
    return state


# ---------------------------------------------------------------------------
# Primary path - semantic seeds + directional raster barriers
# ---------------------------------------------------------------------------

_ROOM_LABEL_ALIASES = {
    "LNDRY": "LAUNDRY",
    "REC ROOM AREA": "REC ROOM AREA",
    "MECHANICAL": "MECHANICAL",
}


def _semantic_raster_rooms(state, image_area, id_gen) -> list[Room]:
    """Extract rooms as free-space components containing room-label OCR.

    Directional opening suppresses text and short symbols. Closing only along
    each line's own axis bridges doors, windows, and fragmented wall faces
    without globally filling room interiors. The structural-core bounds keep
    schedules and title blocks out, while OCR labels select architectural
    regions from the remaining free-space components.
    """
    config = state.config
    binary = (state.binary_cleaned if state.binary_cleaned is not None
              else state.binary_masked if state.binary_masked is not None
              else state.binary)
    core = state.structural_core_mask
    if binary is None or core is None or not state.raw_texts:
        return []

    seeds = _room_label_seeds(state)
    if len(seeds) < config.semantic_room_min_seeds:
        return []

    line_px = max(3, int(config.room_barrier_min_line_px))
    close_px = max(line_px, int(config.room_barrier_gap_close_px))
    thickness = max(1, int(config.room_barrier_thickness_px))

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((1, line_px), np.uint8)
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((line_px, 1), np.uint8)
    )
    horizontal = cv2.morphologyEx(
        horizontal, cv2.MORPH_CLOSE, np.ones((1, close_px), np.uint8)
    )
    vertical = cv2.morphologyEx(
        vertical, cv2.MORPH_CLOSE, np.ones((close_px, 1), np.uint8)
    )
    horizontal = cv2.dilate(horizontal, np.ones((thickness, 1), np.uint8))
    vertical = cv2.dilate(vertical, np.ones((1, thickness), np.uint8))
    barrier = cv2.bitwise_or(horizontal, vertical)

    x, y, w, h = cv2.boundingRect((core > 0).astype(np.uint8))
    if w < 3 or h < 3:
        return []
    crop = barrier[y:y + h, x:x + w].copy()
    border = max(3, thickness * 2)
    cv2.rectangle(crop, (0, 0), (w - 1, h - 1), 255, border)

    free = (crop == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    if count <= 1:
        return []

    by_component: dict[int, tuple[str, float]] = {}
    for label, confidence, sx, sy in seeds:
        cx, cy = sx - x, sy - y
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        component_id = int(labels[cy, cx])
        if component_id == 0:
            continue
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.min_room_area_px:
            continue
        if area > config.max_room_area_frac * float(w * h):
            continue
        previous = by_component.get(component_id)
        if previous is None or confidence > previous[1]:
            by_component[component_id] = (label, confidence)

    rooms: list[Room] = []
    for component_id, (label, confidence) in by_component.items():
        component = (labels == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = config.semantic_room_poly_epsilon_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [
            Point(float(point[0][0] + x), float(point[0][1] + y))
            for point in approx
        ]
        if len(polygon) < 3:
            continue
        rooms.append(Room(
            id=id_gen(), polygon=polygon,
            area=float(stats[component_id, cv2.CC_STAT_AREA]),
            label=label, label_confidence=confidence,
        ))

    logger.info("semantic raster extraction found %d rooms from %d seeds",
                len(rooms), len(seeds))
    return rooms


def build_semantic_plan_mask(state) -> np.ndarray | None:
    """Return a room-derived plan envelope for upstream drafting filtering."""
    if state.image is None:
        return None
    rooms = _semantic_raster_rooms(
        state, float(state.image.shape[0] * state.image.shape[1]), IdGenerator("SEM")
    )
    if len(rooms) < state.config.semantic_room_min_seeds:
        return None
    points = np.array(
        [[int(round(point.x)), int(round(point.y))]
         for room in rooms for point in room.polygon],
        dtype=np.int32,
    )
    if len(points) < 3:
        return None
    mask = np.zeros(state.image.shape[:2], np.uint8)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    cv2.fillPoly(mask, [hull], 255)
    margin = max(0, int(state.config.semantic_plan_margin_px))
    if margin:
        size = 2 * margin + 1
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    return mask


def _room_label_seeds(state) -> list[tuple[str, float, int, int]]:
    config = state.config
    vocab = sorted(config.room_label_vocab, key=len, reverse=True)
    core = state.structural_core_mask
    seeds = []
    for text in state.raw_texts:
        if text.confidence < config.semantic_room_seed_confidence:
            continue
        normalized = " ".join(text.text.upper().replace("/", "/").split())
        match = next((entry for entry in vocab if entry in normalized), None)
        if match is None:
            continue
        cx, cy = int(round(text.center.x)), int(round(text.center.y))
        if not (0 <= cy < core.shape[0] and 0 <= cx < core.shape[1]):
            continue
        if core[cy, cx] == 0:
            continue
        label = _ROOM_LABEL_ALIASES.get(match, normalized)
        seeds.append((label, float(text.confidence), cx, cy))
    return seeds


# ---------------------------------------------------------------------------
# Primary path - planar graph faces
# ---------------------------------------------------------------------------

def _planar_graph_rooms(state: PipelineState, image_area, id_gen):
    config = state.config

    node_of: dict[str, tuple[float, float]] = {
        j.id: (j.point.x, j.point.y) for j in state.junctions
    }
    tol = config.junction_snap_radius_px

    def resolve(p: Point):
        best, best_d = None, tol
        for jid, (x, y) in node_of.items():
            d = ((p.x - x) ** 2 + (p.y - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = jid, d
        return best

    graph = nx.Graph()
    for jid, xy in node_of.items():
        graph.add_node(jid, pos=xy)
    for wall in state.walls:
        a = resolve(wall.centerline.start)
        b = resolve(wall.centerline.end)
        if a is None or b is None or a == b:
            continue
        graph.add_edge(a, b)

    if graph.number_of_edges() == 0:
        return [], False

    graph = _repair_planarity(graph, config)
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        return [], False

    pos = nx.get_node_attributes(graph, "pos")
    faces = _traverse_faces(embedding)

    candidates: list[Polygon] = []
    for face in faces:
        if len(face) < 3:
            continue
        coords = [pos[n] for n in face]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        area = poly.area
        if area < config.min_room_area_px:
            continue
        if area > config.max_room_area_frac * image_area:
            continue
        if any(p.equals(poly) for p in candidates):
            continue
        candidates.append(poly)

    # a component's outer face traces the union outline of its bounded faces;
    # drop any face that strictly contains another face's interior point
    rooms: list[Room] = []
    for poly in candidates:
        is_outer = any(
            other is not poly
            and poly.area > other.area
            and poly.contains(other.representative_point())
            for other in candidates
        )
        if is_outer:
            continue
        exterior = list(poly.exterior.coords)[:-1]  # no repeated first vertex
        rooms.append(Room(
            id=id_gen(), polygon=[Point(float(x), float(y)) for x, y in exterior],
            area=float(poly.area),
        ))
    return rooms, True


def _repair_planarity(graph: nx.Graph, config) -> nx.Graph:
    from shapely.geometry import LineString

    pos = nx.get_node_attributes(graph, "pos")
    synth = IdGenerator("SYN")
    for _ in range(config.planarity_repair_max_iterations):
        is_planar, _ = nx.check_planarity(graph)
        if is_planar:
            return graph
        crossing = _find_crossing(graph, pos)
        if crossing is None:
            return graph
        (a, b), (c, d), point = crossing
        node = synth()
        graph.add_node(node, pos=point)
        pos[node] = point
        graph.remove_edge(a, b)
        graph.remove_edge(c, d)
        for n in (a, b, c, d):
            graph.add_edge(n, node)
    return graph


def _find_crossing(graph, pos):
    from shapely.geometry import LineString

    edges = list(graph.edges())
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            a, b = edges[i]
            c, d = edges[j]
            if {a, b} & {c, d}:
                continue
            l1 = LineString([pos[a], pos[b]])
            l2 = LineString([pos[c], pos[d]])
            inter = l1.intersection(l2)
            if not inter.is_empty and inter.geom_type == "Point":
                return (a, b), (c, d), (inter.x, inter.y)
    return None


def _traverse_faces(embedding: nx.PlanarEmbedding):
    faces = []
    visited: set[tuple] = set()
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            if (v, w) in visited:
                continue
            face = embedding.traverse_face(v, w, mark_half_edges=visited)
            faces.append(face)
    return faces


# ---------------------------------------------------------------------------
# Fallback path - flood fill
# ---------------------------------------------------------------------------

def _floodfill_rooms(state: PipelineState, image_area, id_gen) -> list[Room]:
    config = state.config
    h, w = state.image.shape[:2]
    wall_mask = np.zeros((h, w), np.uint8)
    for wall in state.walls:
        cl = wall.centerline
        cv2.line(
            wall_mask,
            (int(round(cl.start.x)), int(round(cl.start.y))),
            (int(round(cl.end.x)), int(round(cl.end.y))),
            255, max(1, int(round(wall.thickness + config.floodfill_wall_dilation_px))),
        )

    interior = (wall_mask == 0).astype(np.uint8)
    labels = measure.label(interior, connectivity=1)

    rooms: list[Room] = []
    for region in measure.regionprops(labels):
        if region.area < config.min_room_area_px:
            continue
        if region.area > config.max_room_area_frac * image_area:
            continue
        # regions touching the image border belong to the outer space
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == h or maxc == w:
            continue
        component = (labels == region.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = config.room_poly_epsilon_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [Point(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(polygon) < 3:
            continue
        rooms.append(Room(id=id_gen(), polygon=polygon, area=float(region.area)))
    return rooms


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for room in state.rooms:
        pts = np.array([[int(p.x), int(p.y)] for p in room.polygon], np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [pts], (230, 200, 170))
        overlay = cv2.addWeighted(fill, 0.25, overlay, 0.75, 0)
        cv2.polylines(overlay, [pts], True, (127, 127, 127), 2)
    return overlay
