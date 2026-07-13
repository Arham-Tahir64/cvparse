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

### 3. Fall back when an OCR backend cannot initialize

- Issue: backend discovery handled missing imports but allowed model-cache permission, corruption, and native-runtime initialization errors to abort the CLI. This caused 9 test failures and no baseline output under the restricted runtime.
- Hypothesis: isolate initialization errors per backend and continue to the next backend; callers can then use the existing documented no-OCR behavior when none is usable.
- Files: `apps/api/src/vision/cv/ocr_engines.py`, `tests/unit/vision/cv/test_ocr_engines.py`, `PROGRESS.md`
- Commands/tests: focused failover tests (2 passed); full restricted suite (134 passed versus 122 passed/9 failed at baseline); restricted real-plan CLI writing `debug_output/ocr_fallback.{pdf,png}`.
- Result: retained as a reproducibility fix, not an annotation-score claim. With no usable fallback executable, the CLI now completes in 23.9 s using its existing no-OCR path (186 walls after door splits, 4 doors, 59 windows, 1 room) instead of aborting. The authoritative quality artifact remains the PaddleOCR run `debug_output/door_scale.png`.

### Rejected room-topology experiments

- Existing planar graph: 1 room; existing wall-mask flood fill: 0 rooms.
- Morphological closing of the detected wall mask (5-61 px) produced at most 3 regions and was unstable across closure size.
- Closing raw binary linework produced 27-59 large enclosed components because text, fixtures, schedules, and symbols form false faces.
- Decision: rejected. Matching the reference room classes requires a major redesign (semantic wall rasterization/opening closure or a learned room segmentation model), not a generalizable threshold tweak.

### Rejected window-strategy ablation

- Strategy A (inner-line matching) produced all 53 windows; Strategy B (face-gap scanning) produced 0 on this wall graph.
- Disabling Strategy A removed all explicit window detections. Window IoU moved 0.0427 -> 0.0447 because false-positive coloured pixels fell, but foreground IoU regressed 0.1109 -> 0.1058 and actual window recall was lost.
- Decision: rejected. General improvement needs window-frame evidence, exterior/interior wall semantics, or a trained classifier.

## Final result

- Final authoritative output: `debug_output/final_annotation.pdf`
- Final preview: `debug_output/final_annotation.png`
- Final metric report: `evaluation_output/final.json`
- Final detected objects: 181 walls after door splits, 10 doors, 53 windows, 1 unlabeled room, 63 gaps.
- Final full suite with OCR access: 132 passed after annotation/detection changes; final restricted suite including OCR failover tests: 134 passed.

| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| Macro IoU | 0.0309 | 0.0455 | +0.0146 (+47.2%) |
| Macro F1 | 0.0590 | 0.0844 | +0.0254 (+43.1%) |
| Foreground IoU | 0.0833 | 0.1109 | +0.0276 (+33.1%) |
| Foreground F1 | 0.1539 | 0.1997 | +0.0458 (+29.8%) |
| Wall IoU | 0.0634 | 0.1088 | +0.0454 (+71.6%) |
| Wall boundary F1 | 0.1474 | 0.2266 | +0.0792 (+53.7%) |
| Door IoU | 0.0000 | 0.0009 | +0.0009 |
| Window IoU | 0.0311 | 0.0427 | +0.0116 (+37.3%) |
| Room IoU | 0.0291 | 0.0296 | +0.0005 |

## Remaining limitations

- The supplied ground truth is a raster overlay, not vector/object annotations. Connected components are therefore only an object-count proxy, especially for the connected wall network.
- The reference supplies room-class regions but no machine-readable polygons or object IDs; semantic pixel metrics are authoritative, while object matching is approximate.

## Architectural redesign (2026-07-13)

### Why the graph architecture fails

- Room extraction consumes wall centerlines after drafting-line leakage and greedy face pairing. On the real plan this yields 175 fragmented walls, including exterior dimensions, but omits enough true wall continuity that only one planar face exists.
- The existing fallback rasterizes those same corrupted centerlines, so it finds zero enclosed regions. Closing this mask globally is unstable: small kernels leave rooms open and large kernels manufacture faces from symbols.
- OCR already recognizes eight high-confidence room labels inside the structural core, but stage 09 ignores them and stage 10 can only label rooms that stage 09 has already found.

### Proposed replacement

1. Treat high-confidence room-label OCR inside the structural core as semantic seeds.
2. Extract long horizontal and vertical ink runs directly from the masked binary, independently of the corrupted wall graph.
3. Close each orientation along its axis at opening scale, then thicken it across the axis to form structural barriers while suppressing text and short symbols.
4. Label free-space components inside the structural-core bounds and retain the component containing each semantic seed as a room polygon.
5. Use the existing planar/flood paths when fewer than two reliable semantic seeds exist.

This decouples room topology from false dimension walls, closes door/window interruptions without filling room interiors, exports labeled polygons through the existing result renderer/adapter, and creates semantic regions that can later constrain wall and door validation.

### Prototype evidence

- Eight separated labeled rooms: Guest, Bath, Gym, Laundry, Linen, Storage, Mechanical, and Rec Room.
- Room IoU 0.0296 -> 0.5054; room F1 0.0575 -> 0.6715; room boundary F1 0.3719 -> 0.4275.
- Foreground IoU 0.1109 -> 0.5215; macro IoU 0.0455 -> 0.1632.
- Full implementation retained. The raw-input run completed in 362.2 s with
  181 walls, 10 doors, 53 windows, 8/8 labeled rooms, and 63 gaps.
- Full suite: 135 passed. Export verified in
  `debug_output/semantic_rooms_full.{pdf,png}`; metrics in
  `evaluation_output/semantic_rooms_full.json`.
- Final room IoU 0.5050, room F1 0.6711, room boundary F1 0.4274;
  foreground IoU 0.5214 and macro IoU 0.1630.
