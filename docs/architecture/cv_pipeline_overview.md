# CV Pipeline Overview

Implementation of the floor plan parsing pipeline specified in
`.agents/cv_pipeline/`. Code lives at `apps/api/src/vision/cv/`; one test file
per module at `tests/unit/vision/cv/`.

## Flow

`run_pipeline()` (`pipeline.py`) runs modules 01-10 over a shared
`PipelineState`, then `state.to_takeoff_result()` produces `CVTakeoffResult`.
`serialize.py` formats the canonical JSON (schema `1.0.0`) and SVG overlay;
`vision/adapters/annotation_adapter.py` converts to the frontend annotation
document (called by the API route only).

| Stage | File | Output |
|-------|------|--------|
| 01 preprocessing | `preprocessing.py` | `image`, `binary` |
| 02 structural ROI | `structural_roi.py` | `binary_masked`, `structural_roi_mask` |
| 03 line detection | `line_detection.py` | `raw_lines`, `raw_texts` |
| 04 line filtering | `line_filters.py` | `classified_lines` |
| 05 wall extraction | `wall_extraction.py` | `walls` (with provenance) |
| 06 junction snapping | `junction_snapping.py` | `junctions`, split walls |
| 07 door detection | `door_detection.py` | `doors`, door `gaps`, wall splits |
| 08 window detection | `window_detection.py` | `windows`, window `gaps` |
| 09 room extraction | `room_extraction.py` | `rooms` |
| 10 OCR labeling | `ocr_labeling.py` | room labels |
| 11 serialization | `serialize.py` | JSON + SVG |

## Implementation notes beyond the spec

These are places where the spec was ambiguous or self-contradictory and a
choice was made; revisit against real plans.

- **Gap-aware merging (03).** Merge collinear fragments when the gap is
  under `line_merge_gap_tol_px`, OR when the gap region's ink fill is at
  least `line_merge_bridge_fill_min`. The second clause bridges junction
  artifacts wider than the gap tolerance (e.g. an 8 px crossing-wall gap),
  per ARCHITECTURE.md's gap-aware-merging rationale, while door openings
  (near-zero fill) never bridge.
- **Structural ROI component ranking (02).** Candidate components are ranked
  by hole-filled area, not raw seed area, so a hollow plan outline beats a
  smaller solid blob such as a schedule table. The chosen component is
  hole-filled before dilation so the ROI covers the plan interior.
- **Window side-fill (08).** A face gap is accepted only when BOTH sides show
  wall ink. The module spec's algorithm text says reject only when both
  sides are empty, but its test criterion 9 and the stated purpose
  (filtering wall ends/corners) require the stricter rule.
- **Thin-wall stub bypass (05).** `thin_branch_stub_bypass_length_px`
  defaults to 0 (bypass disabled): door-arc chords otherwise register as
  thin walls and the wall-erasure step in module 07 then erases the arc
  before Hough sees it. Genuine junction stubs pass the orthogonal-support
  check anyway.
- **Outer-face filtering (09).** In addition to the 85%-of-image filter, a
  face that strictly contains another face's interior point is dropped -
  the outer face of a multi-room component traces the union outline and is
  otherwise indistinguishable from a room on small sheets.
- **OCR degradation.** Module 03's first pass logs a warning and continues
  with no text when no OCR engine is importable (it only locates text);
  module 10 raises `PipelineError` per its spec. Engines are cached in
  `ocr_engines.py`, shared by both modules.
- **Vocabulary.** `MASTER` and `MASTER BEDROOM` were added to the default
  vocab so the spec's multi-word combination example ("MASTER" + "BEDROOM")
  can actually match.
- **Door Hough tuning.** `hough_circles_param2` defaults to 25 because thin
  (2 px) quarter arcs were consistently missed at 30. The maximum radius is
  160 px: at 1/4" = 1'-0" scale and 200 DPI, a common 2'-6" leaf is ~125 px.
  The debug overlay in module 07 remains the tuning tool for other scales.
- **Dimension-line hardening (added after testing on a real Calgary plan).**
  Root cause of dimension strings detected as walls: with no OCR engine the
  text-based rules 1-3 silently no-op, and stacked parallel dimension rows
  pair as wall faces. Changes: (a) PaddleOCR wrapper supports the 3.x API
  (`predict()`, `enable_mkldnn=False` - the oneDNN backend crashes on some
  Windows CPUs); (b) first-pass OCR runs tiled (`ocr_engines.read_tiled`) so
  large sheets keep small dimension text at native resolution instead of
  being downscaled past legibility (PaddleOCR caps the long side at 4000 px);
  (c) new Rule 0.5 in module 04: thin segment with midpoint outside
  `state.structural_core_mask` (pre-dilation plan component, now stored by
  module 02) -> dimension; (d) Rule 4b: dense parallel non-axis-aligned
  clusters -> hatch regardless of length (kills dropped-ceiling shading on
  Manhattan plans, config `hatch_diagonal_cluster`); (e) tick detection
  accepts 45-degree slash ticks (angle window >= 30 degrees, search radius
  8 px); (f) `dimension_line_max_thickness_px` 1.5 -> 2.5 because LSD reports
  1-4 px widths for hairlines and wall-face strokes alike.
- **Debug-tool error tolerance.** `run_pipeline_state(...,
  tolerate_stage_errors=True)` records a failed stage in debug messages and
  continues; used by the annotate CLI so a `NoRoomsExtractedError` still
  yields an annotated PDF. The API route keeps the spec's fail-loud behavior.

## Annotated-PDF debug output

`annotate_pdf.py` draws every detection onto the plan as a vector overlay
(walls red / orange when low-confidence, doors green, windows brown, door
gaps cyan, window gaps pink, rooms light-blue fill with labels, junctions
blue, structural ROI purple dashed) with a legend in the top-left corner.
For PDF inputs the original page is annotated; raster inputs get a new PDF
page. The CLI runs the whole pipeline and writes the annotated PDF:

```
PYTHONPATH=apps/api/src python -m vision.cv.annotate_cli plan.pdf out.pdf \
    [--page N] [--dpi N] [--param2 25] [--max-arc-radius 160] [--preview out.png]
```

If no OCR engine is installed the labeling stage is skipped with a warning
(`run_pipeline_state(..., skip_stages=...)`) instead of failing. Door arc
radius and accumulator threshold remain CLI options for plans whose scale
differs substantially from the 200-DPI defaults.

## Running

```
python -m venv .venv
.venv/Scripts/pip install -r apps/api/requirements.txt
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/uvicorn api.main:app --app-dir apps/api/src
```

`POST /api/cv/takeoff` accepts a multipart upload (`file`, optional
`mime_type`, `page_number`, `include_annotations`).
