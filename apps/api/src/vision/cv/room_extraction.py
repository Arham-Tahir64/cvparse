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

    rooms: list[Room] = []
    planar_ok = False
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
