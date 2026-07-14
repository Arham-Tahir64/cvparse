"""Module 01 - Raster Preprocessing.

Convert any raster input to a clean, deskewed, binarized grayscale image at
the working DPI. Populates state.image, state.binary, state.dpi.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .models import PipelineState, PipelineError

logger = logging.getLogger("flowbuildr.cv.preprocessing")

MODULE = "01_preprocessing"

_PDF_MIMES = {"application/pdf"}
_RASTER_MIMES = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp",
}


def load_image(
    file_bytes: bytes, mime_type: str, dpi: int = 200, page_number: int | None = 0
) -> np.ndarray:
    """Decode raw upload bytes into a grayscale uint8 ndarray at `dpi`."""
    if mime_type in _PDF_MIMES:
        return _load_pdf(file_bytes, dpi, page_number)
    if mime_type in _RASTER_MIMES:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise PipelineError(MODULE, f"failed to decode image with MIME type {mime_type}")
        return image
    raise PipelineError(MODULE, f"unsupported MIME type: {mime_type}")


def _load_pdf(file_bytes: bytes, dpi: int, page_number: int | None) -> np.ndarray:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            raise PipelineError(MODULE, "PDF contains no pages")
        index = page_number if page_number is not None else _select_best_page(doc, dpi)
        if index >= doc.page_count:
            raise PipelineError(
                MODULE, f"page {index} out of range (PDF has {doc.page_count} pages)"
            )
        return _render_page(doc, index, dpi)
    finally:
        doc.close()


def _render_page(doc, index: int, dpi: int) -> np.ndarray:
    page = doc[index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=__import__("fitz").Matrix(zoom, zoom), colorspace="gray")
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return image[:, :, 0].copy() if image.ndim == 3 else image.copy()


def _select_best_page(doc, dpi: int) -> int:
    """Pick the page with the highest pixel variance (proxy for the plan page)."""
    if doc.page_count == 1:
        return 0
    preview_dpi = 40  # cheap low-res render just for variance ranking
    variances = []
    for i in range(doc.page_count):
        img = _render_page(doc, i, preview_dpi)
        variances.append(float(np.var(img)))
    best = int(np.argmax(variances))
    if variances[best] <= 0:
        logger.warning("no high-variance page found; falling back to page 0")
        return 0
    return best


def _resample_to_dpi(image: np.ndarray, source_dpi: float, target_dpi: int) -> np.ndarray:
    if source_dpi <= 0 or abs(source_dpi - target_dpi) < 1:
        return image
    scale = target_dpi / source_dpi
    new_size = (max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interp)


def _deskew(image: np.ndarray, max_angle_deg: float) -> np.ndarray:
    _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None or len(coords) < 10:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # normalize minAreaRect angle (convention varies by OpenCV version) to [-45, 45]
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90
    if abs(angle) < 1e-3 or abs(angle) > max_angle_deg:
        if abs(angle) > max_angle_deg:
            logger.debug("deskew skipped: detected angle %.2f exceeds limit", angle)
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _binarize(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if config.binarization_method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    block = config.adaptive_block_size
    if block % 2 == 0:
        block += 1
        logger.debug("adaptive_block_size corrected to odd value %d", block)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, config.adaptive_c,
    )


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] <= min_area
    small[0] = False  # background label stays
    out = binary.copy()
    out[small[labels]] = 0
    return out


def run(state: PipelineState) -> PipelineState:
    config = state.config
    if state.image is None:
        if state.source_path is None:
            raise PipelineError(MODULE, "no source_path and no pre-loaded image")
        file_bytes = Path(state.source_path).read_bytes()
        state.image = load_image(
            file_bytes, state.mime_type, config.working_dpi, state.page_number
        )

    image = state.image
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = _deskew(image, config.deskew_max_angle_deg)
    image = cv2.fastNlMeansDenoising(image, h=config.denoise_h)
    binary = _binarize(image, config)
    binary = _remove_small_components(binary, config.min_component_area_px)

    state.image = image
    state.binary = binary
    state.dpi = config.working_dpi

    if config.debug_visualize and config.debug_output_dir:
        out_dir = os.path.join(config.debug_output_dir, "01_preprocessing")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "grayscale.png"), image)
        cv2.imwrite(os.path.join(out_dir, "binary.png"), binary)

    logger.info("preprocessing done: image %sx%s", image.shape[1], image.shape[0])
    return state


def visualize(state: PipelineState, base_image: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    if state.binary is not None:
        overlay[state.binary > 0] = (0, 0, 255)
    return overlay
