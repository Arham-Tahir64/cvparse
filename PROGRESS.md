# Annotation Pipeline Improvement Log

## Baseline

- Date: 2026-07-12
- Commit: `c2d0899` (`main`, equal to `origin/main` before changes)
- Input: `112125_14_ARCH-3.pdf`, page 1, 200 DPI (7200 x 4800 working image)
- Ground truth: untracked `goal.png`, byte-identical to the uploaded screenshot
- Command: `PYTHONPATH=apps/api/src python -m vision.cv.annotate_cli 112125_14_ARCH-3.pdf debug_output/baseline.pdf --preview debug_output/baseline.png`
- Runtime: 365.9 s; line detection 246.0 s, second OCR 96.8 s
- Detected objects: 175 walls, 0 doors, 53 windows, 1 unlabeled room, 53 gaps
- Tests outside the filesystem sandbox: 131 passed, 2 warnings in 17.11 s
- Tests inside the sandbox: 122 passed, 9 failed because PaddleOCR's existing model cache is unreadable to the sandbox account
- Baseline annotation: `debug_output/baseline.pdf`; raster preview: `debug_output/baseline.png`

## Initial issues

1. Wall extraction accepts many short and unsupported fragments (175 walls), so the wall graph does not represent the architectural topology.
2. Door detection has zero recall. The default maximum arc radius (80 px) is below the documented ~125 px radius at this drawing scale, and wall fragmentation further prevents hinge-to-wall association.
3. Window detection produces 53 candidates, far above the reference count, because wall gaps and interior fragments are misclassified as windows.
4. Room extraction collapses the plan to one face rather than the reference room/area regions; OCR consequently labels 0/1 rooms.
5. OCR accounts for about 94% of runtime and its cache permission failure aborts the CLI instead of taking the documented graceful fallback.

## Plan

1. Establish reproducible semantic pixel, boundary, and connected-object metrics using the source PDF to separate coloured overlays from original linework.
2. Correct wall topology first, since doors, windows, and rooms all consume it.
3. Improve doors and windows independently, retaining only measured gains with regression tests.
4. Improve room extraction and labeling after wall/opening topology is stable.
5. Run the complete suite and real-plan evaluation for every retained change; commit and push each focused improvement.

## Iterations

### 0. Reproducible evaluator

- Hypothesis: semantic masks reconstructed from alpha-blended overlay colours provide stable per-class pixel and boundary metrics without requiring hidden vector ground-truth data.
- Files: `.gitignore`, `tools/evaluate_annotation.py`, `PROGRESS.md`
- Commands/tests: `python tools/evaluate_annotation.py 112125_14_ARCH-3.pdf goal.png debug_output/baseline.png --output evaluation_output/baseline.json`; `python -m py_compile tools/evaluate_annotation.py`.
- Result: baseline macro IoU 0.0309, macro F1 0.0590, foreground IoU 0.0833, foreground F1 0.1539.

| Class | IoU | Precision | Recall | F1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Wall | 0.0634 | 0.2694 | 0.0765 | 0.1192 | 0.1474 |
| Door | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Window | 0.0311 | 0.0609 | 0.0597 | 0.0603 | 0.1325 |
| Room | 0.0291 | 0.3775 | 0.0306 | 0.0566 | 0.3782 |

### 1. Render the full measured wall thickness

- Issue: PDF annotations rendered wall strokes at half of `visual_thickness`, although that field and PyMuPDF's stroke width both represent full width. Correct detections therefore covered only half of each reference wall band.
- Hypothesis: rendering the complete measured width should improve wall overlap and boundary agreement without changing detector behavior.
- Files: `apps/api/src/vision/cv/annotate_pdf.py`, `tests/unit/vision/cv/test_annotate_pdf.py`, `PROGRESS.md`
- Commands/tests: focused renderer tests (3 passed); full suite (131 passed); unchanged CLI command writing `debug_output/wall_width.{pdf,png}`; semantic evaluation writing `evaluation_output/wall_width.json`.
- Objects/runtime: unchanged at 175 walls, 0 doors, 53 windows, 1 unlabeled room; 375.2 s.
- Result: retained. Wall IoU 0.0634 -> 0.1095 (+72.7%), wall F1 0.1192 -> 0.1974, wall boundary F1 0.1474 -> 0.2310. Foreground IoU 0.0833 -> 0.1074; macro IoU 0.0309 -> 0.0449 and macro F1 0.0590 -> 0.0831.

### 2. Search the physical door-radius range at working scale

- Issue: the 80 px maximum excluded a documented ~125 px common 2'-6" door leaf at 1/4" scale and 200 DPI; accumulator threshold 30 also missed thin architectural arcs.
- Hypothesis: a 160 px upper bound and the already-tested 25-vote Hough threshold cover common scaled leaves while existing arc-coverage and wall-association checks limit false positives.
- Files: `apps/api/src/vision/cv/config.py`, `apps/api/src/vision/cv/annotate_cli.py`, `tests/unit/vision/cv/test_door_detection.py`, `PROGRESS.md`
- Candidate comparison from one shared stage-6 state: radius 160/threshold 30 -> 3 doors; radius 160/threshold 25 -> 10 doors; additionally increasing wall snap 15 -> 25 -> 14 doors but worse door precision, so the snap change was rejected.
- Commands/tests: full suite (132 passed); downstream candidate harness; full current-default CLI writing `debug_output/door_scale.{pdf,png}`; semantic evaluation writing `evaluation_output/door_scale.json`.
- Objects/runtime: 181 walls (door splits included), 10 doors, 53 windows, 1 unlabeled room, 63 gaps; 374.3 s.
- Result: retained. Door IoU 0 -> 0.0009 and door F1 0 -> 0.0018; foreground IoU 0.1074 -> 0.1109; macro IoU 0.0449 -> 0.0455 and macro F1 0.0831 -> 0.0844. The small pixel score reflects the renderer's sparse hinge/line representation versus filled ground-truth door regions.

## Remaining limitations

- The supplied ground truth is a raster overlay, not vector/object annotations. Connected components are therefore only an object-count proxy, especially for the connected wall network.
- The reference supplies room-class regions but no machine-readable polygons or object IDs; semantic pixel metrics are authoritative, while object matching is approximate.
