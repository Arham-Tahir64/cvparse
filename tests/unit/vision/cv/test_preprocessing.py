"""Tests for module 01 - raster preprocessing, against the spec's test criteria."""
import cv2
import numpy as np
import pytest

from vision.cv import preprocessing
from vision.cv.config import PipelineConfig
from vision.cv.models import PipelineError, PipelineState


def make_state(image=None, **config_kwargs):
    config = PipelineConfig(**config_kwargs)
    state = PipelineState(config=config)
    state.image = image
    return state


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def pdf_bytes(width_in=2.0, height_in=1.5) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=width_in * 72, height=height_in * 72)
    page.draw_rect(fitz.Rect(20, 20, 80, 60), color=(0, 0, 0), width=3)
    data = doc.tobytes()
    doc.close()
    return data


def test_png_load_shape_matches():
    img = np.full((120, 200), 255, np.uint8)
    loaded = preprocessing.load_image(png_bytes(img), "image/png")
    assert loaded.shape == (120, 200)


def test_pdf_load_dimensions_at_200_dpi():
    loaded = preprocessing.load_image(pdf_bytes(2.0, 1.5), "application/pdf", dpi=200)
    assert abs(loaded.shape[1] - 400) <= 2
    assert abs(loaded.shape[0] - 300) <= 2


def test_unsupported_mime_raises():
    with pytest.raises(PipelineError):
        preprocessing.load_image(b"xx", "application/zip")


def _detect_angle(image):
    _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    angle = cv2.minAreaRect(coords)[-1]
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def _rotated_rect_image(angle_deg):
    img = np.full((400, 400), 255, np.uint8)
    cv2.rectangle(img, (100, 150), (300, 250), 0, thickness=4)
    m = cv2.getRotationMatrix2D((200, 200), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (400, 400), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def test_small_rotation_corrected():
    state = make_state(_rotated_rect_image(3.0))
    preprocessing.run(state)
    assert abs(_detect_angle(state.image)) < 1.0


def test_large_rotation_left_unchanged():
    img = _rotated_rect_image(30.0)
    state = make_state(img.copy())
    preprocessing.run(state)
    # 30deg maps to a residual within minAreaRect ambiguity; verify no rotation applied:
    # the deskew stage must leave geometry as-is, so detected angle stays far from 0
    assert abs(_detect_angle(state.image)) > 5.0


def test_binary_polarity():
    img = np.full((100, 100), 230, np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 30, thickness=6)
    state = make_state(img)
    preprocessing.run(state)
    assert state.binary[state.image < 100].max() == 255
    assert set(np.unique(state.binary)) <= {0, 255}
    # some bright pixel maps to 0
    assert (state.binary[state.image > 200] == 0).any()


def test_small_component_removed_large_survives():
    img = np.full((200, 200), 255, np.uint8)
    img[10:12, 10:12] = 0          # 2x2 noise blob
    img[50:100, 50:100] = 0        # 50x50 component
    state = make_state(img)
    preprocessing.run(state)
    assert state.binary[10:12, 10:12].sum() == 0
    # adaptive threshold keeps the outline of a large filled square; the
    # component must survive small-component removal
    region = state.binary[45:105, 45:105]
    assert (region == 255).sum() > 100


def test_load_image_pdf_returns_grayscale_ndarray():
    arr = preprocessing.load_image(pdf_bytes(), "application/pdf")
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.dtype == np.uint8


def test_image_and_binary_dimensions_match():
    img = np.full((150, 220), 255, np.uint8)
    cv2.line(img, (10, 75), (210, 75), 0, 5)
    state = make_state(img)
    preprocessing.run(state)
    assert state.image.shape == state.binary.shape
    assert state.dpi == state.config.working_dpi
