"""Tests for the annotated-PDF renderer."""
import cv2
import fitz
import numpy as np
import pytest

from vision.cv import ocr_engines, pipeline
from vision.cv.annotate_pdf import annotate_image_as_pdf, annotate_pdf_page
from vision.cv.config import PipelineConfig
from vision.cv.models import TextElement


class FakeEngine:
    def read(self, image, confidence_threshold):
        return [TextElement("KITCHEN", (270, 390, 380, 410), 0.95)]


def synthetic_plan():
    img = np.full((800, 1000), 255, np.uint8)
    for a, b in [((150, 150), (850, 150)), ((850, 150), (850, 650)),
                 ((850, 650), (150, 650)), ((150, 650), (150, 150)),
                 ((500, 150), (500, 650))]:
        cv2.line(img, a, b, 0, 8)
    img[399:462, 495:506] = 255
    cv2.ellipse(img, (500, 460), (60, 60), 0, 180, 270, 0, 2)
    cv2.line(img, (500, 460), (440, 460), 0, 2)
    return img


def run_state(monkeypatch):
    monkeypatch.setattr(ocr_engines, "get_engine", lambda *_: FakeEngine())
    return pipeline.run_pipeline_state(
        image=synthetic_plan(), config=PipelineConfig(
            hough_circles_param2=25.0, door_arc_min_radius_px=20.0,
        )
    )


def test_annotate_image_as_pdf(monkeypatch):
    state = run_state(monkeypatch)
    result = state.to_takeoff_result()
    out = annotate_image_as_pdf(
        state.image, result, dpi=200,
        roi_mask=state.structural_roi_mask, junctions=state.junctions,
    )
    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.page_count == 1
    drawings = doc[0].get_drawings()
    assert len(drawings) > len(result.walls)  # walls + rooms + legend etc.
    wall_drawings = [
        drawing for drawing in drawings
        if drawing["fill_opacity"] == pytest.approx(0.85, abs=0.01)
        and (
            drawing["fill"] == pytest.approx((0.84, 0.15, 0.16), abs=0.01)
            or drawing["fill"] == pytest.approx((1.0, 0.5, 0.05), abs=0.01)
        )
    ]
    assert len(wall_drawings) == len(result.walls)
    door_sectors = [
        drawing for drawing in drawings
        if drawing["fill"] == pytest.approx((0.17, 0.63, 0.17), abs=0.01)
        and drawing["fill_opacity"] == pytest.approx(0.24, abs=0.01)
    ]
    assert len(door_sectors) == len(result.doors) == 1
    text = doc[0].get_text()
    assert "walls" in text  # legend
    assert "KITCHEN" in text  # room label
    doc.close()


def test_annotate_pdf_page(monkeypatch):
    # build a PDF from the plan image, then annotate that PDF's page
    img = synthetic_plan()
    src = fitz.open()
    page = src.new_page(width=img.shape[1] * 72 / 200, height=img.shape[0] * 72 / 200)
    ok, buf = cv2.imencode(".png", img)
    page.insert_image(page.rect, stream=buf.tobytes())
    pdf_bytes = src.tobytes()
    src.close()

    state = run_state(monkeypatch)
    result = state.to_takeoff_result()
    out = annotate_pdf_page(
        pdf_bytes, result, dpi=200, page_number=0,
        roi_mask=state.structural_roi_mask, junctions=state.junctions,
    )
    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.page_count == 1
    assert len(doc[0].get_drawings()) > 5
    doc.close()


def test_reconstructed_wall_mask_renders_as_filled_region():
    image = np.full((120, 180), 255, np.uint8)
    wall_mask = np.zeros(image.shape, np.uint8)
    wall_mask[40:80, 30:150] = 255
    from vision.cv.models import CVTakeoffResult

    out = annotate_image_as_pdf(
        image, CVTakeoffResult(), dpi=72, wall_mask=wall_mask,
    )
    doc = fitz.open(stream=out, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()

    # The complete rectangle interior is tinted wall red; surrounding pixels
    # remain the white source rather than receiving a centerline/outline only.
    assert rendered[60, 90, 0] > rendered[60, 90, 1]
    assert np.all(rendered[15, 15] > 245)


def test_semantic_masks_preserve_door_and_window_ownership():
    image = np.full((300, 300), 255, np.uint8)
    door_mask = np.zeros(image.shape, np.uint8)
    window_mask = np.zeros(image.shape, np.uint8)
    door_mask[150:230, 80:180] = 255
    window_mask[185:205, 140:250] = 255
    from vision.cv.models import CVTakeoffResult

    out = annotate_image_as_pdf(
        image, CVTakeoffResult(), dpi=72,
        door_mask=door_mask, window_mask=window_mask,
    )
    doc = fitz.open(stream=out, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()

    # Door-only pixels are green; the later window mask owns their overlap.
    assert int(rendered[165, 100, 1]) - int(rendered[165, 100, 0]) >= 100
    assert rendered[195, 160, 2] > rendered[195, 160, 1]


def test_clean_semantic_export_omits_diagnostic_vectors(monkeypatch):
    state = run_state(monkeypatch)
    result = state.to_takeoff_result()

    verbose = annotate_image_as_pdf(
        state.image, result, dpi=200,
        roi_mask=state.structural_roi_mask, junctions=state.junctions,
        wall_mask=state.wall_mask, door_mask=state.door_mask,
        window_mask=state.window_mask,
        room_instance_mask=state.room_instance_mask,
        include_diagnostics=True,
    )
    clean = annotate_image_as_pdf(
        state.image, result, dpi=200,
        roi_mask=state.structural_roi_mask, junctions=state.junctions,
        wall_mask=state.wall_mask, door_mask=state.door_mask,
        window_mask=state.window_mask,
        room_instance_mask=state.room_instance_mask,
        include_diagnostics=False,
    )

    verbose_doc = fitz.open(stream=verbose, filetype="pdf")
    clean_doc = fitz.open(stream=clean, filetype="pdf")
    assert len(clean_doc[0].get_drawings()) < len(verbose_doc[0].get_drawings())
    assert "KITCHEN" in verbose_doc[0].get_text()
    assert "KITCHEN" not in clean_doc[0].get_text()
    assert "door gaps" in verbose_doc[0].get_text()
    assert "door gaps" not in clean_doc[0].get_text()
    pix = clean_doc[0].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    rendered = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, pix.n,
    )[..., :3]
    # Exact room rasters replace vector polygons in clean mode; no unfinished
    # polygon path may be committed as an opaque black fill.
    assert np.mean(rendered[250:550, 250:750]) > 100
    verbose_doc.close()
    clean_doc.close()


def test_skip_stage_records_message():
    state = pipeline.run_pipeline_state(
        image=synthetic_plan(),
        config=PipelineConfig(
            hough_circles_param2=25.0, door_arc_min_radius_px=20.0,
        ),
        skip_stages=("14_ocr_labeling",),
    )
    assert any("14_ocr_labeling" in m for m in state.debug.messages)
    assert "14_ocr_labeling" not in state.debug.stage_timings
