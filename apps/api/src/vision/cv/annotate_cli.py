"""CLI: run the pipeline on a plan file and write an annotated PDF.

Usage (from repo root):
    .venv/Scripts/python -m vision.cv.annotate_cli INPUT OUTPUT.pdf \
        [--page N] [--dpi N] [--param2 X] [--preview OUT.png]

Runs with PYTHONPATH=apps/api/src or via the repo's conftest path. If no OCR
engine is installed, the labeling stage is skipped with a warning (labels
will be missing; everything else renders).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import fitz

from . import ocr_engines
from .annotate_pdf import annotate_image_as_pdf, annotate_pdf_page
from .config import PipelineConfig
from .pipeline import run_pipeline_state
from .preprocessing import load_image

logger = logging.getLogger("flowbuildr.cv.annotate_cli")

_MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}


def main(argv=None) -> int:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page", type=int, default=0, help="PDF page number")
    parser.add_argument("--dpi", type=int, default=200, help="working DPI")
    parser.add_argument("--param2", type=float, default=defaults.hough_circles_param2,
                        help="Hough accumulator threshold for door arcs (20-50)")
    parser.add_argument("--min-arc-radius", type=float,
                        default=defaults.door_arc_min_radius_px,
                        help="min door swing radius in px")
    parser.add_argument("--max-arc-radius", type=float, default=160.0,
                        help="max door swing radius in px (a 2'-6\" door at "
                             "1/4\" scale and 200 dpi is ~125 px)")
    parser.add_argument("--preview", type=Path, default=None,
                        help="also rasterize the annotated page to this PNG")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="export drafting, cleaned, and semantic masks here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    mime = _MIME_BY_SUFFIX.get(args.input.suffix.lower())
    if mime is None:
        parser.error(f"unsupported input type: {args.input.suffix}")

    config = PipelineConfig(
        working_dpi=args.dpi, hough_circles_param2=args.param2,
        door_arc_min_radius_px=args.min_arc_radius,
        door_arc_max_radius_px=args.max_arc_radius,
        debug_visualize=args.debug_dir is not None,
        debug_output_dir=str(args.debug_dir) if args.debug_dir is not None else None,
    )
    file_bytes = args.input.read_bytes()
    image = load_image(file_bytes, mime, dpi=args.dpi, page_number=args.page)
    logger.info("loaded %s: %dx%d px at %d dpi", args.input.name,
                image.shape[1], image.shape[0], args.dpi)

    skip: tuple[str, ...] = ()
    if ocr_engines.get_engine(config.ocr_engine) is None:
        logger.warning("no OCR engine installed; skipping room labeling")
        skip = ("14_ocr_labeling",)

    start = time.perf_counter()
    state = run_pipeline_state(
        image=image, mime_type=mime, config=config,
        page_number=args.page, skip_stages=skip,
        tolerate_stage_errors=True,  # debug tool: render whatever was detected
    )
    result = state.to_takeoff_result()
    logger.info(
        "pipeline done in %.1fs: %d walls, %d doors, %d windows, %d rooms, %d gaps",
        time.perf_counter() - start, len(result.walls), len(result.doors),
        len(result.windows), len(result.rooms), len(result.gaps),
    )

    if mime == "application/pdf":
        annotated = annotate_pdf_page(
            file_bytes, result, dpi=args.dpi, page_number=args.page,
            roi_mask=state.structural_roi_mask, junctions=state.junctions,
            wall_mask=state.wall_mask,
            door_mask=state.door_mask, window_mask=state.window_mask,
            room_instance_mask=state.room_instance_mask,
        )
    else:
        annotated = annotate_image_as_pdf(
            state.image, result, dpi=args.dpi,
            roi_mask=state.structural_roi_mask, junctions=state.junctions,
            wall_mask=state.wall_mask,
            door_mask=state.door_mask, window_mask=state.window_mask,
            room_instance_mask=state.room_instance_mask,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(annotated)
    logger.info("wrote %s", args.output)

    if args.preview is not None:
        doc = fitz.open(stream=annotated, filetype="pdf")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pix.save(str(args.preview))
        doc.close()
        logger.info("wrote preview %s", args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
