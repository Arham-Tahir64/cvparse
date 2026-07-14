"""Module 03 - Line Detection.

LSD on the ROI-masked binary, gap-aware collinear merging, and a first-pass
OCR to locate text regions for module 04.
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np

from . import ocr_engines
from .geometry import unit_direction
from .models import LineSegment, NoLinesDetectedError, PipelineState, Point, TextElement

logger = logging.getLogger("flowbuildr.cv.line_detection")

MODULE = "03_line_detection"


def run(state: PipelineState) -> PipelineState:
    config = state.config
    binary = state.binary_masked if state.binary_masked is not None else state.binary

    segments = _detect_segments(binary, config)
    segments = [s for s in segments if s.length >= config.min_line_length_px]
    if not segments:
        raise NoLinesDetectedError(MODULE, "no line segments after length filtering")

    state.debug.segment_counts["03_raw_lsd"] = len(segments)
    merged = _merge_collinear(segments, binary, config)
    state.debug.segment_counts["03_merged"] = len(merged)
    logger.debug("LSD segments: %d, after merge: %d", len(segments), len(merged))

    state.raw_lines = merged

    # The clean structural pass reuses OCR from the proposal pass. OCR must see
    # the original plan so labels and measurement context remain available.
    if not state.raw_texts:
        engine = ocr_engines.get_engine(config.ocr_engine)
        if engine is None:
            msg = "first-pass OCR skipped: no OCR engine available"
            logger.warning(msg)
            state.debug.messages.append(msg)
            state.raw_texts = []
        else:
            executor = None
            if ocr_engines.supports_worker_pool(engine) and config.ocr_parallel_workers > 1:
                executor = ocr_engines.get_executor(
                    engine.name, config.ocr_parallel_workers,
                    config.ocr_worker_cpu_threads,
                )
            # Second-pass OCR (module 10) depends only on state.image, which
            # no later stage mutates. Submit it to a dedicated single-worker
            # pool (engine-default threads: the full-sheet read is the longest
            # single job) so it runs while modules 04-13 work without slowing
            # the tile pool.
            if state.ocr_second_pass_future is None and executor is not None:
                state.ocr_second_pass_future = ocr_engines.submit_read(
                    engine.name, 1, state.image
                )
            state.raw_texts = run_ocr_first_pass(
                state.image, engine, config.ocr_first_pass_confidence,
                executor=executor,
            )

    if config.debug_visualize and config.debug_output_dir:
        os.makedirs(config.debug_output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(config.debug_output_dir, "03_lines.png"),
            visualize(state, state.image),
        )
    return state


def run_ocr_first_pass(image, engine, confidence_threshold, executor=None) -> list[TextElement]:
    """Locate text regions. This pass finds text, it does not read labels.

    Tiled so large sheets keep small dimension text at native resolution.
    """
    texts = ocr_engines.read_tiled(engine, image, confidence_threshold, executor=executor)
    logger.debug("first-pass OCR found %d text elements", len(texts))
    return texts


# ---------------------------------------------------------------------------
# LSD
# ---------------------------------------------------------------------------

def _detect_segments(binary: np.ndarray, config) -> list[LineSegment]:
    # LSD expects dark lines on light background
    inverted = cv2.bitwise_not(binary)
    lines, widths = _run_lsd(inverted, config)
    segments = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        thickness = float(widths[i]) if widths is not None else 1.0
        segments.append(LineSegment(
            start=Point(float(x1), float(y1)), end=Point(float(x2), float(y2)),
            thickness=thickness, id=f"L{i:05d}",
        ))
    return segments


def _run_lsd(image: np.ndarray, config):
    try:
        lsd = cv2.createLineSegmentDetector(
            refine=cv2.LSD_REFINE_ADV, scale=config.lsd_scale,
            sigma_scale=config.lsd_sigma_scale,
        )
        lines, widths, _, _ = lsd.detect(image)
        if lines is None:
            return [], None
        return lines.reshape(-1, 4), (widths.ravel() if widths is not None else None)
    except cv2.error:
        logger.warning("OpenCV LSD unavailable; falling back to pylsd-nova")
        from pylsd import lsd as pylsd_detect

        result = pylsd_detect(image.astype(np.float64))
        if result is None or len(result) == 0:
            return [], None
        return result[:, :4], result[:, 4]


# ---------------------------------------------------------------------------
# Gap-aware collinear merging
# ---------------------------------------------------------------------------

def _merge_collinear(
    segments: list[LineSegment], binary: np.ndarray, config
) -> list[LineSegment]:
    angle_tol = math.radians(config.line_merge_angle_tol_deg)
    n_bins = max(1, int(round(math.pi / angle_tol)))

    angle_groups: dict[int, list[LineSegment]] = {}
    for seg in segments:
        # round (not floor) so angles just under pi wrap into the 0 bin
        b = int(round(seg.angle_rad / angle_tol)) % n_bins
        angle_groups.setdefault(b, []).append(seg)

    merged: list[LineSegment] = []
    for b, group in angle_groups.items():
        merged.extend(_merge_angle_group(group, binary, config))
    return merged


def _merge_angle_group(group: list[LineSegment], binary: np.ndarray, config):
    # subgroup by perpendicular offset from a shared reference axis
    ref_angle = group[0].angle_rad
    ux, uy = math.cos(ref_angle), math.sin(ref_angle)
    nx, ny = -uy, ux
    tol = config.line_merge_perpendicular_tol_px

    offset_groups: dict[int, list[LineSegment]] = {}
    for seg in group:
        mid = seg.midpoint
        offset = mid.x * nx + mid.y * ny
        offset_groups.setdefault(int(round(offset / tol)), []).append(seg)

    out: list[LineSegment] = []
    for subgroup in offset_groups.values():
        out.extend(_merge_chain(subgroup, (ux, uy), binary, config))
    return out


def _merge_chain(
    subgroup: list[LineSegment], direction: tuple[float, float], binary, config
) -> list[LineSegment]:
    ux, uy = direction

    def proj(p: Point) -> float:
        return p.x * ux + p.y * uy

    items = []
    for seg in subgroup:
        lo, hi = sorted((proj(seg.start), proj(seg.end)))
        items.append([lo, hi, seg])
    items.sort(key=lambda it: it[0])

    out: list[LineSegment] = []
    cur_lo, cur_hi, cur = items[0]
    cur_members = [cur]
    for lo, hi, seg in items[1:]:
        gap = lo - cur_hi
        # tiny gaps merge outright; larger gaps only when the gap region is
        # backed by wall ink (junction artifact), never across empty door
        # openings (ARCHITECTURE: gap-aware merging)
        if gap <= config.line_merge_gap_tol_px or (
            _gap_fill_fraction(cur_members[-1], seg, binary)
            >= config.line_merge_bridge_fill_min
        ):
            cur_hi = max(cur_hi, hi)
            cur_members.append(seg)
        else:
            out.append(_build_merged(cur_members, cur_lo, cur_hi, direction))
            cur_lo, cur_hi, cur_members = lo, hi, [seg]
    out.append(_build_merged(cur_members, cur_lo, cur_hi, direction))
    return out


def _gap_fill_fraction(a: LineSegment, b: LineSegment, binary: np.ndarray) -> float:
    """Fraction of foreground pixels sampled along the gap between a's end and b's start.

    The caller supplies the same proposal or cleaned binary used by LSD, so a
    removed drafting line cannot bridge segments back together.
    """
    # endpoints closest to each other define the gap
    pairs = [(pa, pb) for pa in (a.start, a.end) for pb in (b.start, b.end)]
    pa, pb = min(pairs, key=lambda pq: pq[0].distance_to(pq[1]))
    gap_len = pa.distance_to(pb)
    if gap_len < 1:
        return 1.0
    n = max(3, int(gap_len))
    ts = np.linspace(0.0, 1.0, n)
    xs = np.rint(pa.x + ts * (pb.x - pa.x)).astype(int)
    ys = np.rint(pa.y + ts * (pb.y - pa.y)).astype(int)
    h, w = binary.shape[:2]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not valid.any():
        return 0.0
    return float((binary[ys[valid], xs[valid]] > 0).mean())


def _build_merged(
    members: list[LineSegment], lo: float, hi: float, direction: tuple[float, float]
) -> LineSegment:
    if len(members) == 1:
        return members[0]
    ux, uy = direction
    # anchor: average perpendicular offset of members
    nx, ny = -uy, ux
    offset = sum(m.midpoint.x * nx + m.midpoint.y * ny for m in members) / len(members)
    start = Point(lo * ux + offset * nx, lo * uy + offset * ny)
    end = Point(hi * ux + offset * nx, hi * uy + offset * ny)
    return LineSegment(
        start=start, end=end,
        thickness=max(m.thickness for m in members),
        id="+".join(m.id for m in members if m.id) or members[0].id,
    )


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    for seg in state.raw_lines:
        cv2.line(
            overlay,
            (int(seg.start.x), int(seg.start.y)),
            (int(seg.end.x), int(seg.end.y)),
            (0, 0, 255), 1,
        )
    return overlay
