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
        if drawing["color"] == pytest.approx((0.84, 0.15, 0.16), abs=0.01)
        and drawing["width"] != pytest.approx(3.0, abs=0.01)  # legend swatch
    ]
    expected_widths = sorted(max(0.8, wall.visual_thickness * 72 / 200)
                             for wall in result.walls)
    actual_widths = sorted(drawing["width"] for drawing in wall_drawings)
    assert actual_widths == pytest.approx(expected_widths, abs=0.01)
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


def test_skip_stage_records_message():
    state = pipeline.run_pipeline_state(
        image=synthetic_plan(),
        config=PipelineConfig(
            hough_circles_param2=25.0, door_arc_min_radius_px=20.0,
        ),
        skip_stages=("10_ocr_labeling",),
    )
    assert any("10_ocr_labeling" in m for m in state.debug.messages)
    assert "10_ocr_labeling" not in state.debug.stage_timings
