"""Module 02 - Structural ROI.

Find the floor plan body via its own wall structure and mask everything
outside it into state.binary_masked. No positional assumptions about title
blocks or legends.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .models import PipelineState

logger = logging.getLogger("flowbuildr.cv.structural_roi")

MODULE = "02_structural_roi"


def run(state: PipelineState) -> PipelineState:
    config = state.config
    binary = state.binary

    # Step 1 - wall seed extraction via directional opening
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.roi_h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.roi_v_kernel_len))
    h_seed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_seed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    seed = cv2.bitwise_or(h_seed, v_seed)

    # Step 2 - close seed to connect wall fragments
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.roi_close_kernel_px, config.roi_close_kernel_px)
    )
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, close_kernel)

    # Step 3 - select the largest sufficiently big component
    image_area = binary.shape[0] * binary.shape[1]
    min_area = config.roi_min_component_area_frac * image_area
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    # rank candidates by hole-filled area so a hollow plan outline beats a
    # solid but smaller non-plan blob (e.g. a schedule table)
    best_label, best_area, best_filled = 0, 0, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        component = np.where(labels == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component, contours, -1, 255, thickness=cv2.FILLED)
        filled_area = int((component > 0).sum())
        if filled_area > best_area:
            best_label, best_area, best_filled = i, filled_area, component

    if best_label == 0:
        msg = "structural ROI: no wall-seed component found; falling back to full image"
        logger.warning(msg)
        state.debug.messages.append(msg)
        mask = np.full_like(binary, 255)
        state.structural_core_mask = mask.copy()
    else:
        mask = best_filled  # component with interior holes filled
        # tight pre-dilation wall-mass coverage, kept for module 04
        state.structural_core_mask = mask.copy()
        # Step 4 - expand ROI to include wall faces, arcs, and openings
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.roi_dilate_kernel_px, config.roi_dilate_kernel_px)
        )
        mask = cv2.dilate(mask, dilate_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Step 5 - apply mask
    state.binary_masked = cv2.bitwise_and(binary, mask)
    state.structural_roi_mask = mask
    state.debug.roi_area_fraction = float((mask > 0).sum()) / float(image_area)

    if config.debug_visualize and config.debug_output_dir:
        out_dir = os.path.join(config.debug_output_dir, "02_structural_roi")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "binary_before.png"), binary)
        cv2.imwrite(os.path.join(out_dir, "binary_masked.png"), state.binary_masked)
        cv2.imwrite(os.path.join(out_dir, "roi_overlay.png"), visualize(state, state.image))

    logger.info("structural ROI area fraction: %.3f", state.debug.roi_area_fraction)
    return state


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    if state.structural_roi_mask is not None:
        contours, _ = cv2.findContours(
            state.structural_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 0, 255), 3)
    return overlay
