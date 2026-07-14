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

### Drafting-line redesign proposal

- Failure: the structural-core component is connected to exterior dimension
  strings and extension rows. LSD width is also unreliable (roughly 1-4 px
  for both hairlines and wall faces), so 1,257 segments remain `unknown` and
  151 parallel pairs become walls.
- Replacement: construct a semantic plan envelope from the convex hull of
  seeded room regions, dilated by a wall margin. Classify segments whose
  midpoints lie outside that envelope as drafting annotations before wall
  pairing. Lines inside the envelope still pass through the existing
  text/tick/hatch logic, preserving legitimate thin interior walls.
- Retained implementation: semantic margin ablation tested 50, 90, 130, and
  180 px; 90 px best preserved exterior walls while excluding drafting rows.
- Full run: 364.6 s, 151 walls, 9 doors, 37 windows, 8 labeled rooms, 46 gaps;
  full suite 137 passed.
- Classification: `unknown` 1,257 -> 906; `dimension` 28 -> 566.
- Wall false-positive pixels 12,161 -> 6,965; precision 0.2717 -> 0.3726;
  IoU 0.1089 -> 0.1135; boundary F1 0.2257 -> 0.2754.
- Foreground IoU 0.5214 -> 0.5326. Door IoU moved 0.0012 -> 0.0010
  because one marginal detection was removed; the structural gain was retained.
- Output: `debug_output/semantic_refilter_full.{pdf,png}`; metrics:
  `evaluation_output/semantic_refilter_full.json`.

### Door redesign proposal

- Failure: the circle-first stage accepts any partial circular ink near a wall.
  On the retained plan its nine detections are toilet/sink outlines, window
  markers, or annotations; the visible door swings are mostly absent. The
  detector also stores the midpoint of an observed arc as the leaf endpoint,
  splits the wall only at that point, and the PDF renderer draws a full circle.
  Consequently the door export contains only 131 coloured pixels versus 7,553
  reference pixels (IoU 0.0010), even before accounting for localization error.
- Replacement: keep Hough only as a proposal generator, then require
  door-specific evidence: a physical leaf-radius range, quarter-arc endpoints
  aligned parallel/perpendicular to the supporting wall, thick-wall
  continuation on the hinge side, a structural opening toward the jamb, and
  radial leaf ink. Retain the observed arc samples as explicit geometry.
- Export/rendering: serialize hinge, leaf endpoint, and sampled swing arc;
  render the bounded quarter-swing sector rather than a full circle; and add
  door/window elements to the frontend annotation adapter so detections are not
  lost after the CV result stage.
- Expected effect: fixtures and drafting circles fail the wall-opening test,
  true swings gain area and boundary overlap, and downstream consumers receive
  the same door geometry that is evaluated in the PDF output.

### Door redesign retained

- Implementation: Hough is now proposal-only. Candidates must pass a
  scale-aware circumference check, cardinal endpoint alignment, supporting-wall
  continuation, low structural support through the opening, and radial leaf
  support. Accepted doors retain 18 ordered swing-arc samples and a confidence
  score. The PDF/SVG exporters draw a bounded translucent swing sector, and the
  frontend adapter now exports both doors and windows.
- Files: `apps/api/src/vision/cv/{config.py,door_detection.py,models.py,annotate_pdf.py,serialize.py}`,
  `apps/api/src/vision/adapters/annotation_adapter.py`,
  `tools/evaluate_annotation.py`, door/renderer/serialization/end-to-end tests,
  and `PROGRESS.md`.
- Rejected first attempt: fixed +/-1 px circumference sampling yielded zero
  real-plan doors. A radius-proportional 2-6 px annulus retained the semantic
  tests while recovering anti-aliased PDF arcs.
- Tests/commands: focused door/serialization/renderer tests; updated synthetic
  end-to-end plan with a real wall opening and radial leaf; full suite 140
  passed. Cached stage-6 comparison produced 9 doors. The authoritative raw
  CLI run completed in 366.2 s with 148 walls, 12 doors, 37 windows, 8/8
  labeled rooms, and 49 gaps.
- Retained result: door IoU 0.0010 -> 0.1059; door F1 0.0021 -> 0.1916;
  door recall 0.0011 -> 0.3119; foreground IoU 0.5326 -> 0.5623; macro IoU
  0.1648 -> 0.1880. Wall IoU moved 0.1135 -> 0.1112 and room IoU 0.5071 ->
  0.4934 because filled door sectors replace overlapping wall/room pixels in
  the mutually exclusive evaluator; the substantially larger door and overall
  gains justify retention.
- Output: `debug_output/door_semantic_full.{pdf,png}`; metrics:
  `evaluation_output/door_semantic_full.json`.

## Current result after the redesigns

| Metric | Original baseline | Current | Change |
|---|---:|---:|---:|
| Macro IoU | 0.0309 | 0.1880 | +0.1571 |
| Foreground IoU | 0.0833 | 0.5623 | +0.4790 |
| Wall IoU | 0.0634 | 0.1112 | +0.0478 |
| Door IoU | 0.0000 | 0.1059 | +0.1059 |
| Window IoU | 0.0311 | 0.0415 | +0.0104 |
| Room IoU | 0.0291 | 0.4934 | +0.4643 |

Remaining limitations: the raster reference has no object IDs and its
connected-component object counts fragment anti-aliased overlays. One detected
double-leaf/closet swing remains outside the reference door class, and several
nearby central doors overlap. Separating those cases reliably requires door
type/schedule semantics or labeled examples rather than a plan-specific tag
rule. Window localization and wall recall remain materially below the room and
door gains.

## Dedicated drafting-removal redesign (continued objective)

### Completion audit of the previous architecture

- The semantic plan envelope only reclassified LSD segments during line
  filtering. It did not remove drafting pixels, and line detection, wall face
  support, door detection, window detection, and room barriers still consumed
  `binary_masked`, which contains the measurement layer.
- No drafting mask or cleaned plan image was exported. The debug outputs also
  lacked independent wall, window, and room-region masks.
- PDF walls were stroked centerlines and windows were always drawn as
  horizontal ticks, even on vertical supporting walls. The latter cannot
  represent the complete window span and wall relationship.

### Proposed replacement

1. Run the existing OCR/line classification as a proposal pass, before final
   structural detection.
2. Build a dedicated drafting mask from dimension text, dimension/tick/leader/
   grid/hatch segments, connected extension geometry, and all ink outside the
   OCR-seeded plan envelope.
3. Protect provisional parallel wall faces, remove only the drafting mask, and
   directionally repair short gaps inside protected wall corridors. Door-sized
   openings and curved swing geometry remain untouched.
4. Re-run line detection and classification on the cleaned binary, then run
   wall, door, window, and room stages exclusively on that cleaned input.
5. Export the drafting mask, cleaned image, structural-protection mask, filled
   wall mask, window mask, room-region mask, and a consistent combined class
   mask. Render wall footprints as filled polygons and windows as filled spans
   aligned to their supporting wall.

This is a two-pass, context-aware preprocessing architecture rather than a
global morphology filter: geometry proposes removal, semantic location and
measurement text supply context, and paired structural faces explicitly veto
destructive removal.

### Dedicated drafting removal retained

- Implementation: the pipeline now runs proposal OCR/lines and contextual
  classification, writes an explicit drafting mask, removes it while restoring
  139 high-confidence paired wall corridors, and re-runs line detection and
  classification on the cleaned binary. The clean pass reduced remaining
  dimension-classified segments from 566 to 3 and removed 94,256 net ink
  pixels without modifying the source image used by OCR.
- Safety evidence: synthetic tests prove that a dimension baseline and
  dimension text are removed, paired wall faces survive, curved door geometry
  survives, exterior drafting ink is removed, and only short protected gaps
  are repaired. On the real plan, wall IoU, room IoU, and door IoU all improved
  rather than regressed.
- Full-input command: `python -m vision.cv.annotate_cli
  112125_14_ARCH-3.pdf debug_output/drafting_cleaned_full.pdf --preview
  debug_output/drafting_cleaned_full.png --debug-dir
  debug_output/drafting_cleaned_full_intermediates` with
  `PYTHONPATH=apps/api/src`.
- Full run: 362.0 s; 124 walls, 4 doors, 7 windows, 8/8 labeled rooms, and 11
  gaps. Full suite: 148 passed.

### Filled wall and scale-aware window masks retained

- Walls now render and export as filled face-to-face polygons using the larger
  of measured and paired-face thickness. Windows render after walls as filled
  blue spans aligned to their supporting wall; the independent wall mask is
  cut out wherever the window mask owns the footprint.
- Window candidates require seeded-room exterior/hull context. The physical
  search ceiling increased from 220 to 340 px because a common 5'-0" window is
  about 250 px at the configured 1/4" scale and 200 DPI. This recovered two
  correct bottom windows that the old ceiling categorically excluded.
- Exported intermediates:
  `05_drafting_removal/{drafting_mask,structural_protection_mask,cleaned_binary,cleaned_image}.png`
  and `13_semantic_masks/{wall_mask,door_mask,window_mask,room_region_mask,combined_class_mask}.png`.

| Metric | Previous final | Drafting-cleaned final | Change |
|---|---:|---:|---:|
| Macro IoU | 0.1880 | 0.2664 | +0.0784 |
| Macro F1 | 0.2831 | 0.3965 | +0.1134 |
| Foreground IoU | 0.5623 | 0.6088 | +0.0465 |
| Wall IoU | 0.1112 | 0.1987 | +0.0875 |
| Wall boundary F1 | 0.2587 | 0.3345 | +0.0758 |
| Door IoU | 0.1059 | 0.1233 | +0.0174 |
| Window IoU | 0.0415 | 0.1954 | +0.1539 |
| Room IoU | 0.4934 | 0.5481 | +0.0547 |

- Final output: `debug_output/drafting_cleaned_full.{pdf,png}`.
- Metric report: `evaluation_output/drafting_cleaned_full.json`.
- Remaining differences: the seed-only room stage still lacks an explicit
  stair/circulation polygon; only two of the three lower exterior windows are
  well localized; several top/right window proposals are false; four detected
  doors do not cover every reference swing; and wall recall is 0.2530. Further
  reliable separation of those cases needs additional plan examples or
  learned/schedule-aware class evidence rather than coordinates from this
  reference image.

## Structure-aware wall-region reconstruction (continued objective)

### Failure analysis before the architectural change

- The current semantic wall mask is not reconstructed from wall boundaries.
  It draws one thick OpenCV centerline for every detected `Wall`. Paired walls
  are truncated to the overlap of two LSD segments, each segment can be used
  by only one pair, and independent strokes are never joined topologically.
  This clips corners and T-junctions and loses a long wall whenever either
  face is fragmented or missing.
- The single-face fallback has no estimate of which side contains the wall
  mass. Its line width is reused as the complete wall thickness, so it cannot
  recover the region between an observed face and an absent opposite face.
- On the retained full run the reference contains 29,500 wall pixels at
  evaluation resolution, while the generated annotation contains 15,535.
  Recall is 0.2530. Of the 22,036 false-negative pixels, 49.8% lie within
  8 px and 67.7% within 16 px of a predicted wall, which is strong evidence of
  clipped thickness/endpoints; the remaining large components are whole
  missing perimeter/interior runs, not a color-rendering artifact.

### Proposed replacement

1. Preserve detected wall boundaries as a dedicated intermediate mask instead
   of reducing them immediately to independent centerline strokes.
2. Rasterize paired faces as closed quadrilateral polygons spanning the actual
   face-to-face interval. Extend polygon ends only where a nearby collinear or
   orthogonal wall supplies junction support.
3. Build a room-boundary topology scaffold from the already detected semantic
   free-space regions. Admit scaffold pixels only where directional structural
   ink or a paired wall corridor supplies evidence, preventing furniture and
   open-room boundaries from becoming walls.
4. Repair the union with directional, thickness-bounded gap closure and local
   junction hulls. Explicit door and window ownership masks are subtracted
   after reconstruction so openings remain intact.
5. Export detected boundaries, reconstructed polygons, repaired wall mask,
   and final combined annotation, and retain the redesign only if wall recall
   and IoU improve without a disproportionate precision or opening regression.

This replaces stroke rendering with evidence-gated region reconstruction; it
does not use global dilation or any coordinate from the reference annotation.

### Wall-region reconstruction retained

- The semantic stage now exports the detected source boundaries, clean-pass
  filled wall polygons, a repaired wall mask, a separately inferred exterior
  ring, and the final combined annotation. The repaired mask unions only two
  independent structural signals: final wall polygons and high-confidence
  paired corridors from the proposal pass. A 21 px directional close repairs
  only gaps already bracketed by that union.
- In Manhattan mode, proposal corridors must contain a 161 px horizontal or
  vertical run. This is deliberately greater than twice the configured 80 px
  maximum wall thickness, so a diagonal drafting band's cross-section cannot
  survive while short clean-pass wall polygons remain intact.
- The exterior pass uses room polygons as observed inner faces and scans the
  cleaned binary outward for sustained parallel wall evidence on every side.
  The full input measured left/right/top/bottom offsets of 67/34/35/67 px.
  The second-largest supported thickness reconstructs the weak one-sided
  faces as a consistent 67 px shell, bounded by the configured maximum. This
  filled the exterior corners and runs without dilating interior walls.
- A structural-core rectangularity gate (0.85 minimum; 0.900 on this input)
  prevents that complete rectangular shell from being inferred on L-shaped
  plans; those inputs retain polygon/corridor reconstruction only.
- Door opening lines, the complete detected swing-sector mask, and window
  spans are subtracted after reconstruction and gap repair. The PDF renderer
  consumes the resulting RGBA wall-region mask instead of recreating partial
  centerline polygons, so opening holes and fragmented contours survive the
  export.

### Iterations and rejected changes

- Allowing a long LSD face to pair with multiple disjoint fragments increased
  cached wall IoU from 0.1987 to 0.2348, but changed wall erasure/context and
  regressed door IoU to 0.1138 and window IoU to 0.1381. That upstream change
  was reverted; reconstruction is post-detection and preserves the proven
  object proposals.
- Unioning proposal corridors without Manhattan run filtering reached wall
  IoU 0.3308 but retained a diagonal stair/drafting band. A 31 px directional
  opening was still shorter than the band's cross-section. The thickness-
  bounded 161 px gate removed it and increased the authoritative wall IoU to
  0.5398.
- Using only independently measured exterior offsets reached wall IoU 0.4197.
  The shell-consistency rule recovered weak top/right faces and reached 0.5398
  while also improving door, room, foreground, and macro metrics.

### Authoritative full-input result

- Commands: full suite `python -m pytest -q`; full CLI `python -m
  vision.cv.annotate_cli 112125_14_ARCH-3.pdf
  debug_output/wall_regions_full.pdf --preview
  debug_output/wall_regions_full.png --debug-dir
  debug_output/wall_regions_full_intermediates`; evaluator
  `python tools/evaluate_annotation.py 112125_14_ARCH-3.pdf goal.png
  debug_output/wall_regions_full.png --output
  evaluation_output/wall_regions_full.json`.
- Tests: 152 passed. Raw pipeline: 358.5 s, 124 walls, 4 doors, 7 windows,
  8/8 labeled rooms, and 11 gaps.

| Metric | Previous final | Wall-region final | Change |
|---|---:|---:|---:|
| Wall precision | 0.4805 | 0.6068 | +0.1263 |
| Wall recall | 0.2530 | 0.8301 | +0.5771 |
| Wall F1 | 0.3315 | 0.7011 | +0.3696 |
| Wall IoU | 0.1987 | 0.5398 | +0.3411 |
| Wall boundary F1 | 0.3345 | 0.5637 | +0.2292 |
| Door IoU | 0.1233 | 0.1277 | +0.0044 |
| Window IoU | 0.1954 | 0.2013 | +0.0059 |
| Room IoU | 0.5481 | 0.5513 | +0.0032 |
| Foreground IoU | 0.6088 | 0.6979 | +0.0891 |
| Macro IoU | 0.2664 | 0.3550 | +0.0886 |

- Residual audit: the wall class has 24,489 true-positive, 15,866 false-
  positive, and 5,011 false-negative pixels at evaluation resolution. Of the
  remaining misses, 51.1% are within 3 px and 70.2% within 8 px of the repaired
  mask; the largest missed component is 590 px. There is no longer a large
  missing exterior run. Overfill is concentrated around dense laundry/storage
  cabinetry, short top-room partitions, and the mechanical-room interior.
  More permissive corridor recovery raised those false positives, while local
  erosion reduced wall coverage, so neither was retained.
- Outputs: `debug_output/wall_regions_full.{pdf,png}` and
  `evaluation_output/wall_regions_full.json`. Required intermediates are under
  `debug_output/wall_regions_full_intermediates/13_semantic_masks/` as
  `wall_boundaries.png`, `wall_polygons.png`, `exterior_wall_ring.png`,
  `wall_repaired_mask.png`, `wall_mask.png`, `combined_class_mask.png`, and
  `wall_error_vs_goal.png`.

## Interior drafting and local-thickness redesign (in progress)

### Reproducible baseline and failure decomposition

- Preserved baseline: commit `8cff946`, full result
  `evaluation_output/wall_regions_full.json` (wall IoU 0.5398, precision
  0.6068, recall 0.8301, boundary F1 0.5637). The working tree contained only
  the user's untracked `.claude/` directory and `goal.png`; neither will be
  modified or committed.
- A cached replay of the same detected state was saved only in ignored debug
  output. At native mask resolution, the exterior-ring component is reliable
  (precision 0.8248, IoU 0.5665), but the clean-pass wall polygons have only
  0.4852 precision and the proposal protection corridors only 0.5271.
- Proposal corridors contribute 5,965 exclusive false-positive pixels. They
  were designed as a permissive cleanup veto, not as a semantic wall detector;
  exporting them converts protected dimension/leader geometry into wall area.
- Paired-face separations are bimodal: the lower structural mode is roughly
  6--35 px, while numerous 52--74 px pairs span unrelated parallel drafting
  lines. Rendering the larger of face separation and visual thickness makes
  these false pairs into very broad walls. The 17 single-face fallbacks have
  only 49 true-positive versus 195 false-positive pixels on this input.
- Controlled ablation, with no reference coordinates or labels used by the
  pipeline, shows that excluding cleanup-only proposal corridors from semantic
  export and bounding interior widths at 25 px improves direct wall IoU from
  0.5323 to 0.5929 and precision from 0.591 to 0.708 (recall 0.843 to 0.785).

### Proposed replacement before implementation

1. Keep structural-protection corridors solely in the drafting-cleanup stage;
   semantic wall export will use final detected wall evidence plus the
   separately evidence-gated exterior shell.
2. Estimate a plan-specific interior wall-width ceiling from the lower,
   consistent mode of paired-face separations. Apply it per interior wall,
   preserving the independently measured exterior shell rather than globally
   dilating or eroding the class mask.
3. Add topology and room-boundary support for ambiguous single-face/floating
   candidates so thin legitimate partitions remain eligible while isolated
   measurement rules do not automatically become walls.
4. Export a dedicated interior drafting/measurement mask in addition to the
   combined drafting mask, so the removal decision and residual failure mode
   are directly inspectable.
5. Re-run unit tests, cached comparison, and the complete raw pipeline. Keep
   the redesign only if wall precision/IoU and visual drafting separation
   improve without material room, door, window, or corridor regression.

### Cleanup/semantic separation and local width retained

- Cleanup-only protection corridors are no longer unioned into the semantic
  wall class. Final wall regions now use clean-pass detections plus the
  independently supported exterior shell. This removes the largest source of
  false wall area without changing upstream room, door, or window proposals.
- Interior polygon width is bounded by a plan-specific estimate: the 40th
  percentile of valid paired-face distances with 15% tolerance. It evaluates
  to about 25 px on this plan, but adapts to the observed structural mode on
  other scales and plans. The exterior ring keeps its independently measured
  67 px shell; there is no global dilation or erosion.
- A dedicated `interior_drafting_mask` is now part of pipeline state and is
  exported in both drafting and semantic debug stages. It contains 457,012
  native-resolution pixels on the full run, separately from exterior notes,
  schedules, and dimension strings in the combined 1,096,858-pixel drafting
  mask.
- Tests: 153 passed. Valid raw run with PaddleOCR: 385.3 s, 124 walls, 4
  doors, 7 windows, 8/8 labeled rooms, and 11 gaps. A sandboxed run that could
  not read the OCR cache was discarded and is not used below.
- Full-run commands: `python -m pytest -q`; `python -m
  vision.cv.annotate_cli 112125_14_ARCH-3.pdf
  debug_output/interior_width_full.pdf --preview
  debug_output/interior_width_full.png --debug-dir
  debug_output/interior_width_full_intermediates`; evaluator `python
  tools/evaluate_annotation.py 112125_14_ARCH-3.pdf goal.png
  debug_output/interior_width_full.png --output
  evaluation_output/interior_width_full.json`.

| Metric | Previous full | Local-width full | Change |
|---|---:|---:|---:|
| Wall precision | 0.6068 | 0.7272 | +0.1204 |
| Wall recall | 0.8301 | 0.7815 | -0.0486 |
| Wall F1 | 0.7011 | 0.7534 | +0.0523 |
| Wall IoU | 0.5398 | 0.6043 | +0.0645 |
| Wall boundary F1 | 0.5637 | 0.5727 | +0.0090 |
| Door IoU | 0.1277 | 0.1372 | +0.0095 |
| Window IoU | 0.2013 | 0.2046 | +0.0033 |
| Room IoU | 0.5513 | 0.5546 | +0.0033 |
| Macro IoU | 0.3550 | 0.3752 | +0.0202 |
| Foreground IoU | 0.6979 | 0.6732 | -0.0247 |

- Inspected full-run intermediates: combined and interior-only drafting masks,
  cleaned binary, source wall boundaries, locally bounded wall polygons,
  exterior ring, and final repaired wall mask. The exterior shell remains
  complete and room/door/window geometry is preserved. Residual false wall
  runs are now narrow and concentrated around storage/stair dimensions,
  mechanical-room leaders, and a few unknown parallel clean-pass pairs. The
  next iteration will test whether inferred room-boundary participation can
  reject these floating rules without discarding legitimate thin partitions.

### Room-boundary wall support retained

- The semantic stage now builds a tolerance band around inferred room
  boundaries and requires wall candidates to participate in that band. This
  is deliberately a post-detection semantic veto: door, window, room, and wall
  object proposals remain available, while floating parallel measurement
  rules no longer automatically enter the exported wall pixels.
- The veto activates only with at least two inferred rooms. Plans with sparse
  or no room evidence retain the geometry-only wall fallback. A focused test
  verifies that a legitimate 6 px partition adjoining two rooms survives and
  a same-width floating rule crossing room interiors is rejected.
- Rejected geometry is exported as `rejected_wall_candidates.png` and unioned
  into the semantic-stage `interior_drafting_mask.png`. The full run rejected
  61 of 122 non-diagonal wall candidates (183,407 native pixels); the updated
  interior drafting mask contains 616,383 pixels.
- Sensitivity check: nine combinations spanning 15--25 px boundary tolerance
  and 0.15--0.25 overlap all improved precision relative to no contextual
  veto. The default 20 px/0.20 setting was retained as the scale-consistent,
  moderate choice; the reference-best direct-mask setting removed nine more
  candidates for only +0.0033 direct IoU and was not adopted as a
  reference-specific optimization.
- Tests: 154 passed. Complete raw run with PaddleOCR: 387.8 s, 124 walls, 4
  doors, 7 windows, 8/8 labeled rooms, and 11 gaps. Output:
  `debug_output/room_supported_full.{pdf,png}`; metrics:
  `evaluation_output/room_supported_full.json`; intermediates:
  `debug_output/room_supported_full_intermediates/`.

| Metric | Local-width full | Room-supported full | Change |
|---|---:|---:|---:|
| Wall precision | 0.7272 | 0.8034 | +0.0762 |
| Wall recall | 0.7815 | 0.7369 | -0.0446 |
| Wall F1 | 0.7534 | 0.7687 | +0.0153 |
| Wall IoU | 0.6043 | 0.6243 | +0.0200 |
| Wall boundary F1 | 0.5727 | 0.5652 | -0.0075 |
| Door IoU | 0.1372 | 0.1373 | +0.0001 |
| Window IoU | 0.2046 | 0.2048 | +0.0002 |
| Room IoU | 0.5546 | 0.5548 | +0.0002 |
| Macro IoU | 0.3752 | 0.3803 | +0.0051 |
| Foreground IoU | 0.6732 | 0.6536 | -0.0196 |

- Visual audit: long false horizontal dimension walls through storage, the
  stair/circulation zone, and the mechanical room are removed. The complete
  exterior ring, labeled room fills, door sectors, window spans, and narrow
  partitions bordering detected rooms remain. Residual differences are now
  dominated by room coverage (unlabeled stair/circulation void), missed door
  swings, and short fragmented interior wall ends rather than broad wall
  over-expansion. Fixing those reliably requires either broader free-space
  seeding/training data or a different learned door/room segmentation model.

## Highlighted room-barrier and missing-wall iteration (in progress)

### Cycle 1 baseline and proposed redesign

- The two screenshots were registered to the 1536x1024 evaluation frame at
  correlation 0.929 and 0.831. Their evaluation crops are rec-room
  `(418,625)-(905,827)` and bath/linen/mechanical strip
  `(741,331)-(841,680)`; native 200-DPI crops are exported under
  `debug_output/focus_baseline/` with the seven requested views.
- Rec-room baseline: room IoU 0.6121, recall 0.6167. The cleaned structural
  binary still contains 637 pixels on the long vertical dimension rule and
  2,471 on the horizontal dimension run. The seeded room terminates exactly
  on the vertical rule, while the goal continues across it.
- Missing-wall baseline: wall IoU 0.2632 and recall 0.3008. The clean-pass
  boundary mask contains the paired faces, but the final room-boundary filter
  rejects the collinear structural chain around linen and mechanical rooms.
- Root cause 1: dimension recognition uses distance to the *midpoint* of a
  line. A 634 px vertical rule only 7 px from its rotated `17' 3/"` OCR box is
  68 px from the line midpoint and remains `unknown`; directional room
  morphology then promotes it to a barrier. Drafting pixels inside provisional
  protection can also be restored before room segmentation.
- Root cause 2: semantic wall admission relies on room-boundary overlap alone.
  Real paired segments such as the 204 px bathroom wall and adjoining
  154--279 px wall chain have strong paired faces and junctions but little
  overlap because the incomplete room segmentation is itself used as the
  support signal.

Proposed coupled change:

1. Recognize dimension rules by their minimum geometric distance to a nearby
   dimension OCR box plus matching horizontal/vertical text orientation,
   rather than midpoint distance. Do not restore pixels explicitly classified
   as drafting; use the existing bounded directional repair to close any small
   holes where a removed rule crosses a protected real wall.
2. Recompute seeded rooms from that repaired clean binary, verifying that a
   dimension line cannot create a second free-space component or truncate the
   labeled room.
3. Keep room-boundary support as the primary semantic wall signal, then run a
   bounded topology-recovery pass for rejected paired walls whose width fits
   the plan-specific structural mode and whose endpoints connect to accepted
   walls/exterior shell or a multi-wall junction. This restores enclosure
   chains without globally relaxing the drafting filter.
4. Gate retention on simultaneous improvement in both registered crops and
   whole-image metrics; reject any cleanup widening that trades room recovery
   for missing structural walls.

### Cycle 1 retained: exact room support and topology recovery

- Long dimension baselines are now associated with the nearest edge of a
  parallel OCR measurement box instead of the line midpoint. A separate
  confirmed-measurement context mask prevents adjacent/continued dimension
  strokes from being exported as semantic walls without deleting nearby
  structural pixels.
- Drafting removal suppresses only text-confirmed measurement pixels inside
  permissive wall protection. Its directional repair closes the small holes
  where a measurement crosses paired real wall faces. A test verifies that
  open-room portions remain removed while both wall faces survive.
- Room extraction now retains the exact selected connected-component raster.
  Simplified outer contours previously filled structural holes whenever a
  recreation/circulation component wrapped around enclosed rooms; the exact
  raster is used for wall-boundary support and final room masks.
- Rejected paired wall candidates can now be recovered through a bounded,
  iterative topology pass when their width matches the plan-specific wall
  mode, confidence is at least 0.80, and both endpoints (or one endpoint plus
  a multi-wall junction) attach to accepted structure. Floating single lines
  remain rejected.
- Rejected experiment: suppressing every drafting pixel inside wall
  protection caused all interior rooms to merge and all 104 wall candidates
  to fail room support. Rejected experiment: widening a global measurement
  corridor raised the missing-wall crop but reduced recreation-crop wall IoU
  to 0.674. Neither change was retained.
- Tests: 159 passed. Full uncached run: 376.1 s, 116 walls, 4 doors, 7 windows,
  8/8 rooms. Output `debug_output/highlight_cycle1_full.{pdf,png}`;
  intermediates `debug_output/highlight_cycle1_full_intermediates/`; focused
  crops `debug_output/focus_cycle1_full/`.

| Focus metric | Baseline | Cycle 1 | Change |
|---|---:|---:|---:|
| Rec-room vertical barrier pixels | 637 | 0 | -637 |
| Rec-room horizontal barrier pixels | 2,471 | 2 | -2,469 |
| Rec-room room IoU | 0.6176 | 0.8718 | +0.2542 |
| Rec-room wall IoU | 0.7174 | 0.7169 | -0.0005 |
| Missing-wall crop wall IoU | 0.2668 | 0.4528 | +0.1860 |
| Missing-wall crop wall boundary F1 | 0.2859 | 0.7394 | +0.4535 |
| Missing-wall crop room IoU | 0.3657 | 0.6618 | +0.2961 |

| Full rendered metric | Previous full | Cycle 1 | Change |
|---|---:|---:|---:|
| Wall IoU | 0.6243 | 0.6359 | +0.0116 |
| Door IoU | 0.1373 | 0.1386 | +0.0013 |
| Window IoU | 0.2048 | 0.1985 | -0.0063 |
| Room IoU | 0.5548 | 0.7425 | +0.1877 |
| Macro IoU | 0.3803 | 0.4289 | +0.0486 |
| Foreground IoU | 0.6536 | 0.8280 | +0.1744 |

### Cycle 2 retained: complete a near-rectangular room corner

- Whole-output inspection found one remaining diagonal truncation across the
  recreation room. The OCR-derived semantic plan hull is 93.0% rectangular,
  but its diagonal lower-right edge clips valid free space.
- Rejected experiment: rectangularizing the upstream plan mask recovered the
  corner but changed line classification, reducing window IoU from 0.2510 to
  0.1332 and recreation-crop wall IoU from 0.7169 to 0.6636.
- Current design keeps the conservative plan hull for drafting, walls, and
  window exterior-context inference. Only the room export receives the
  missing pixels of a >=90%-filled Manhattan rectangle; exact pre-completion
  free-space remains the structural wall-support signal, and original room
  polygons remain the window-context signal.
- Cached and full results retain identical structural detections to Cycle 1:
  116 walls, 4 doors, 7 windows, 8 rooms, and 11 gaps. Recreation-crop room
  IoU improves 0.8718 -> 0.9158; rendered full room IoU improves
  0.7425 -> 0.7744; foreground IoU improves 0.8280 -> 0.8514; macro IoU
  improves 0.4289 -> 0.4356. Wall IoU remains 0.6359. Door IoU changes
  0.1386 -> 0.1344 and window IoU 0.1985 -> 0.1979 because the larger room
  overlay changes rendered colour ownership; direct door/window masks are
  unchanged.
- Tests: 160 passed. Full uncached run: 366.0 s. Final output for this cycle:
  `debug_output/highlight_cycle2_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle2_full.json`; intermediates:
  `debug_output/highlight_cycle2_full_intermediates/`; focused crops:
  `debug_output/focus_cycle2_full/`.

### Cycle 3 goal: door geometry and coverage

- Whole-output inspection after Cycle 2 identifies doors as the largest
  remaining labeled-class error: rendered IoU 0.1344, recall 0.2894, and only
  4 detected doors against substantially more ground-truth swing regions.
- Next hypothesis: the current circle/quarter-arc proposal path is too strict
  for fragmented PDF swing arcs and assigns sectors from incomplete arc
  samples without enough jamb/wall-opening evidence. The next cycle will
  measure each detected door against the truth, audit rejected arc candidates,
  and test whether wall-gap seeded arc assembly can improve recall without
  promoting fixtures or room symbols.

### Cycle 3 retained: preserve door-leaf evidence through structural cleanup

- Root cause: door validation used the drafting-cleaned raster for every
  feature. That is correct for wall continuity and opening evidence, but the
  same cleanup can erase a legitimate thin door leaf. One audited proposal had
  0.00 cleaned leaf support versus 0.35 in the original in-plan raster and was
  therefore rejected before geometry export.
- Door validation now uses two evidence streams. Wall continuation and opening
  checks remain on the cleaned structural raster; only radial leaf support is
  measured on the original binary clipped to the semantic plan. This recovers
  leaves without allowing exterior dimensions to manufacture door openings.
- A measured snap-radius sweep at opening support 0.52 selected 70 px: direct
  door IoU was 0.1756 / 0.2137 / 0.2116 / **0.2234** / 0.2037 / 0.2037 for
  55 / 60 / 65 / 70 / 75 / 80 px. The 75-80 px variants were rejected because
  distant fixture/arc candidates raised false positives by 1,034 pixels.
- A second sweep selected maximum opening support 0.52 as the first stable
  threshold: 0.42 produced 7 doors at direct IoU 0.1980; 0.48 produced 8 at
  0.1903; 0.52-0.60 produced 9 at 0.2234. The existing uninterrupted-wall
  rejection test remains in force, and a new regression test verifies original
  leaf evidence can survive structural cleanup.
- The two registered structural cases are unchanged: recreation room wall IoU
  0.7169, room IoU 0.9158, barrier pixels 0 vertical / 2 horizontal; missing-
  wall crop wall IoU 0.4528, boundary F1 0.7394, room IoU 0.6593. Thus door
  recovery does not trade away measurement rejection or real-wall retention.
- Full rendered results versus Cycle 2: wall IoU 0.6359 -> 0.6386, door IoU
  0.1344 -> 0.1529, window IoU 0.1979 -> 0.1868, room IoU 0.7744 -> 0.7752,
  macro IoU 0.4356 -> 0.4384, foreground IoU 0.8514 -> 0.8528. The window
  detector mask is unchanged (direct IoU 0.2510); its rendered decline is a
  colour-ownership interaction with the five additional door overlays and is
  the next cycle's rendering/overlap issue.
- Tests: 161 passed. Full uncached run: 381.5 s, 116 walls, 9 doors, 7 windows,
  8/8 rooms, and 16 gaps. Commands: `python -m pytest -q`;
  `python -m vision.cv.annotate_cli ...`; `tools/evaluate_annotation.py ...`;
  `debug_output/evaluate_focus_regions.py`. Output:
  `debug_output/highlight_cycle3_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle3_full.json`; intermediates:
  `debug_output/highlight_cycle3_full_intermediates/`; seven-layer crops:
  `debug_output/focus_cycle3_full/`.

### Cycle 4 goal: eliminate semantic overlay ownership regressions

- Whole-output comparison shows direct window geometry did not change in Cycle
  3, while rendered window IoU fell 0.0111 after additional doors were drawn.
  Inspect PDF/preview draw order and semantic ownership so overlapping door,
  window, wall, and room classes are exported consistently with the reference,
  without changing detector masks or either highlighted structural result.

### Cycle 4 retained: export exact, exclusive door/window masks

- Root cause: semantic-mask reconstruction produced the masks used for direct
  evaluation and wall cut-outs, but PDF export discarded the door/window
  rasters and independently rebuilt approximate vector bands and sectors.
  Translucent class overlays also compounded over room fills, making ownership
  ambiguous even when the underlying masks were exclusive.
- PDF and image annotation APIs now accept the final door/window masks. The
  renderer inserts those masks at native plan resolution, retains vector door
  hinges/leaves for readability, and uses window-last exclusivity by clearing
  window-owned pixels from the door raster before export. The previous vector
  path remains the fallback when masks are unavailable.
- Rejected variant: exact masks with the old 0.24 door-sector opacity improved
  macro IoU only to 0.4406 (door 0.1563, window 0.1923). Opaque semantic door
  ownership improved both classes further without changing geometry: door
  0.1707 and window 0.1983.
- Full rendered results versus Cycle 3: wall IoU 0.6386 -> 0.6385, door IoU
  0.1529 -> 0.1707, window IoU 0.1868 -> 0.1983, room IoU 0.7752 -> 0.7756,
  macro IoU 0.4384 -> 0.4458, foreground IoU 0.8528 -> 0.8548. The 0.0001
  wall change is raster antialiasing (2 pixels); structural masks and detector
  counts are identical.
- Both registered cases remain exact: recreation room wall IoU 0.7169, room
  IoU 0.9158, barriers 0 vertical / 2 horizontal; missing-wall crop wall IoU
  0.4528, boundary F1 0.7394, room IoU 0.6593.
- Tests: 162 passed, including an overlap-ownership renderer regression test.
  Full uncached run: 371.7 s, 116 walls, 9 doors, 7 windows, 8/8 rooms, 16
  gaps. Output: `debug_output/highlight_cycle4_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle4_full.json`; intermediates:
  `debug_output/highlight_cycle4_full_intermediates/`; seven-layer crops:
  `debug_output/focus_cycle4_full/`.

### Cycle 5 goal: reduce remaining door-sector geometry error

- Door remains the lowest-overlap class at IoU 0.1707 despite 9 plausible
  detections. Audit each sector against the reference to distinguish hinge,
  radius, quadrant, and duplicate-sector errors. Retain only a geometrically
  explainable correction that raises door IoU without reducing wall, window,
  room, or either highlighted-region metric.

### Cycle 5 retained: resolve contradictory door/window classifications

- Per-sector audit found two zero-overlap door sectors. A simple composite
  confidence gate at 0.60 removed one and raised direct door IoU 0.2234 ->
  0.2319, but rendered wall IoU regressed 0.6385 -> 0.6354 because its wall
  opening was real. That experiment was reverted.
- Quadrant audit showed the other weak sector was generated from a fixture arc
  48 px away from a valid opening. Mirroring it would add 193 true door pixels,
  but the source arc supports the current quadrant, so a goal-driven mirror was
  explicitly rejected as non-generalizable.
- The remaining zero-overlap sector was uniquely co-located with an independently
  detected exterior framed window: its window center is within both 0.65x the
  alleged door radius and 0.75x the detected window width. No other door/window
  pair satisfies both constraints. A physical opening is now prevented from
  being exported as both classes when that strong conflict occurs.
- Conflict resolution removes the contradictory door object and diagnostic gap
  but retains its structural opening separately. Semantic wall reconstruction
  continues to subtract that opening/footprint, avoiding the wall regression;
  the window owns the exported class. Two tests cover conflict resolution and
  structural-opening retention without a door sector.
- Full rendered results versus Cycle 4: wall IoU 0.6385 -> 0.6389, door IoU
  0.1707 -> 0.1763, window IoU 0.1983 -> 0.1990, room IoU 0.7756 -> 0.7783,
  macro IoU 0.4458 -> 0.4481. Direct door IoU improves 0.2234 -> 0.2361 and
  precision 0.5616 -> 0.6521. Foreground IoU changes 0.8548 -> 0.8544 while
  every semantic class IoU improves; this is a 0.0004 ownership-boundary shift.
- Both highlighted cases remain exact: recreation room wall IoU 0.7169, room
  IoU 0.9158, barriers 0 vertical / 2 horizontal; missing-wall crop wall IoU
  0.4528, boundary F1 0.7394, room IoU 0.6593.
- Tests: 164 passed. Full uncached run: 390.9 s, 116 walls, 8 exported doors,
  7 windows, 8/8 rooms, 15 gaps. Output:
  `debug_output/highlight_cycle5_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle5_full.json`; intermediates:
  `debug_output/highlight_cycle5_full_intermediates/`; seven-layer crops:
  `debug_output/focus_cycle5_full/`.

### Cycle 6 goal: reassess the largest remaining whole-image mismatch

- Door remains the lowest class at IoU 0.1763, with one known fixture-derived
  mirrored sector and incomplete recall. Further rule-based quadrant changes
  are unsafe without source-supported arc evidence. Inspect whether a wall-gap-
  seeded detector can recover missing hinges from opening plus leaf evidence;
  otherwise document the need for labeled door examples or a learned detector.

### Cycle 6 retained: require exterior-window shell tangency

- Individual window audit found two structural false positives with zero truth
  overlap: a horizontal framed line at the vertical shell and a vertical framed
  line at the horizontal shell. The previous exterior check required only
  inside/outside room context and hull proximity; it did not verify that the
  supporting wall followed the nearest exterior edge.
- Exterior window candidates must now be parallel to the nearest semantic hull
  edge within 20 degrees. Cross-class door/window reconciliation intentionally
  runs before this filter, so a non-tangent framed candidate can still suppress
  a contradictory door classification before both false classes are removed.
- Direct metrics versus Cycle 5: window IoU 0.2510 -> 0.2624 and wall IoU
  0.5010 -> 0.5027. Full rendered metrics: wall IoU 0.6389 -> 0.6407, door IoU
  0.1763 -> 0.1764, window IoU 0.1990 -> 0.2049, room IoU 0.7783 -> 0.7782,
  macro IoU 0.4481 -> 0.4501. Direct room geometry is unchanged; the 0.0001
  rendered room change is class-overlay ownership.
- Rejected door architecture experiment: using original in-plan ink for circle
  proposals recovered candidates near two missing doors but generated 810
  circles and 23 accepted sectors. Direct door IoU fell 0.2361 -> 0.1852.
  Candidates in the missing-door regions still selected fixture-supported wrong
  quadrants; mirroring them toward the goal would be image-specific.
- Rejected window experiment: requiring a physical two-face wall gap cannot
  distinguish survivors. All five remaining framed candidates have both-open
  fraction 0.0 and either-open fraction 0.0 in the cleaned structure, including
  both high-quality and residual false candidates.
- Both highlighted cases remain exact: recreation room wall IoU 0.7169, room
  IoU 0.9158, barriers 0 vertical / 2 horizontal; missing-wall crop wall IoU
  0.4528, boundary F1 0.7394, room IoU 0.6593.
- Tests: 165 passed. Full uncached run: 378.4 s, 116 walls, 8 doors, 5 windows,
  8/8 rooms, 13 gaps. Final output:
  `debug_output/highlight_cycle6_full.{pdf,png}`; final metrics:
  `evaluation_output/highlight_cycle6_full.json`; final intermediates:
  `debug_output/highlight_cycle6_full_intermediates/`; final seven-layer crops:
  `debug_output/focus_cycle6_full/`.

### Final stopping condition and limitations

- The two user-registered regressions are fixed without a tradeoff: interior
  dimensions no longer split the recreation room, and the missing-wall crop is
  substantially restored while measurement pixels remain rejected.
- Meaningful deviations remain, chiefly door sector localization/quadrants
  (rendered IoU 0.1764), incomplete/misaligned windows (0.2049), local wall
  omissions/excess thickness (0.6407), and imperfect room boundaries (0.7782).
- Repeated original/cleaned Hough, confidence, quadrant, snap, opening-support,
  and wall-face experiments show that fixtures and true doors/windows share the
  same available geometric cues in this drawing. The single provided reference
  cannot supply enough labeled variation to learn that distinction without
  overfitting. Further material improvement requires multiple labeled floor
  plans and a trained semantic/instance detector (or vector/CAD layer metadata),
  followed by the existing topology and semantic-ownership post-processing.

### Cycle 7 goal: export stable semantic room classes and recover circulation

- The latest visual comparison exposed a separate room-architecture failure:
  all room regions were exported with one generic blue fill, and the central
  circulation area was not a room object at all. The extractor allowed only one
  semantic label per connected free-space component. The `UP` label was absent
  from the room vocabulary and its OCR centre landed on hatch/text ink, while
  circulation and recreation space remained connected through an open passage.
- The replacement architecture normalizes labels to stable architectural room
  classes, snaps an obstructed seed only to nearby free space, permits multiple
  semantic seeds in a connected component, and partitions shared Manhattan
  space along the dominant seed axis. Exact per-instance ownership is retained
  as a raster through semantic-mask and PDF export; polygon export remains a
  fallback. This is label- and topology-driven and contains no coordinates or
  pixels from the supplied goal.
- Room completion is recomputed after cleaned structural extraction and clipped
  to the detected interior extent, preventing drafting-context margin from
  becoming recreation-room ownership. Nine independent masks are now exported:
  guest suite, bath 4, gym/yoga, laundry, linen, mechanical, storage,
  stair/circulation, and recreation room.
- Full rendered results versus Cycle 6: wall IoU 0.6407 -> 0.6374, door IoU
  0.1764 -> 0.2077, window IoU 0.2049 -> 0.2190, room IoU 0.7782 -> 0.7789,
  macro IoU 0.4501 -> 0.4607, foreground IoU 0.8543 -> 0.8674. The small wall
  overlay-colour shift is outweighed by improvements in all three other classes
  and does not alter the direct structural mask.
- In the measurement-line crop, room IoU improves 0.9158 -> 0.9393 while wall
  IoU remains 0.7169 and cleaned barriers remain 0 vertical / 2 horizontal.
  In the missing-wall crop, wall IoU remains 0.4528 with boundary F1 0.7394,
  while room IoU improves 0.6593 -> 0.6618. Both registered failure cases
  therefore improve or remain structurally exact at the same time.
- Files changed: `room_extraction.py`, `room_classes.py`, `config.py`,
  `models.py`, `semantic_masks.py`, `annotate_pdf.py`, `annotate_cli.py`, the
  evaluator palette, and focused unit tests. Commands: targeted pytest, full CV
  pytest, full `annotate_cli`, whole-image evaluator, focused-region evaluator,
  and seven-layer crop exporter.
- Tests: 170 passed. Full uncached run: 385.7 s, 116 walls, 8 doors, 5 windows,
  9/9 labeled rooms, and 13 gaps. Output:
  `debug_output/highlight_cycle7_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle7_full.json`; intermediates:
  `debug_output/highlight_cycle7_full_intermediates/`; seven-layer crops:
  `debug_output/focus_cycle7_full/`.

### Cycle 8 goal: align structural-wall rendering without losing topology

- Whole-output comparison now shows class-palette and wall-geometry deviation
  as the largest high-confidence visual mismatch. The direct wall structure is
  unchanged in Cycle 7, but the exported red wall style differs from the goal's
  purple semantic wall ownership and local bands remain too thick or absent.
  Audit renderer style separately from structural geometry, then inspect only
  source-supported face pairs for the remaining local wall error. Retain a
  change only if whole-image wall IoU and both highlighted regions do not
  regress.

### Cycle 8 retained: separate semantic annotation from diagnostics

- A wall-palette-only experiment changed the rendered wall to the reference
  purple/opacity but was rejected: wall IoU fell 0.6374 -> 0.6268, macro IoU
  0.4607 -> 0.4587, and foreground IoU 0.8674 -> 0.8647 because the purple wall
  became ambiguous with the gym room class. Detector geometry was unchanged and
  the experiment was fully reverted.
- The subsequent wall error audit found a broader export fault: even when exact
  semantic rasters were supplied, the renderer added structural ROI outlines,
  gap boxes, junction circles, room polygon outlines, duplicate OCR labels, and
  door hinge/leaf vectors. These are diagnostic evidence, not semantic classes,
  and generated many false objects absent from the goal.
- The PDF/image API now has an explicit `include_diagnostics` mode. Programmatic
  callers retain the detailed diagnostic renderer by default, while the CLI's
  final annotation uses clean semantic mode: exact room, wall, door, and window
  masks plus a semantic-only legend. Diagnostic masks and the requested
  seven-layer crops remain available as separate files.
- The first uncached render exposed an unfinished room-polygon path that
  PyMuPDF committed as a black fill in clean mode. That output was rejected,
  clean mode now skips redundant vector room paths entirely, and a pixel-level
  regression test verifies no opaque black room fill can recur.
- Full rendered results versus Cycle 7: wall IoU 0.6374 -> 0.6465, door IoU
  0.2077 -> 0.2323, window IoU 0.2190 -> 0.2919, room IoU 0.7789 -> 0.7999,
  macro IoU 0.4607 -> 0.4927, foreground IoU 0.8674 -> 0.8734. Predicted
  rendered objects fall from 21 -> 7 walls, 178 -> 55 doors, and 132 -> 7
  windows because debug vectors no longer masquerade as semantic detections.
- Direct masks and both registered structural cases are unchanged: measurement
  crop wall IoU 0.7169, room IoU 0.9393, barriers 0 vertical / 2 horizontal;
  missing-wall crop wall IoU 0.4528, boundary F1 0.7394, room IoU 0.6618.
- Files changed: `annotate_pdf.py`, `annotate_cli.py`, renderer tests, and this
  log. Commands: clean/diagnostic cached render comparison, targeted renderer
  tests, full CV tests, full uncached `annotate_cli`, whole-image evaluator,
  focused evaluator, and crop exporter.
- Tests: 171 passed. Full uncached run: 378.4 s with the same 116 walls, 8
  doors, 5 windows, 9/9 labeled rooms, 13 gaps, 654,286 wall pixels, and
  59,796 window pixels as Cycle 7. Output:
  `debug_output/highlight_cycle8_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle8_full.json`; intermediates:
  `debug_output/highlight_cycle8_full_intermediates/`; seven-layer crops:
  `debug_output/focus_cycle8_full/`.

### Cycle 9 goal: improve remaining door geometry without debug artifacts

- Door is again the lowest-overlap semantic class at IoU 0.2323. Audit the
  eight exact exported sectors against reference door ownership, distinguishing
  true localization/quadrant error from palette/opacity effects. Retain only a
  source-supported geometry or class-ownership change that improves the full
  annotation without regressing walls, windows, rooms, or either structural
  crop.

### Cycle 9 not retained: door geometry evidence remains ambiguous

- Per-sector scoring found strong doors (precision 0.924, 0.997, and 0.875),
  several partial sectors, and one zero-overlap fixture-derived sector. Some
  alternate quadrants overlap the goal better, but the original arc/leaf ink
  does not consistently support those flips: hinge-centred radial support is
  absent for one strong door and prefers the wrong side for another. A quadrant
  rule based on the supplied goal would therefore be image-specific.
- A goal-matched pink/0.62 door palette was rejected because door IoU fell
  0.2323 -> 0.2220 and macro IoU fell 0.4927 -> 0.4913 through ambiguity with
  the mechanical-room fill.
- A topology-safe room-ownership sweep clipped sectors to dilated detected room
  space at 0/5/10/20/30 px, then moved the clipping before wall subtraction and
  swept 20/30/40/50/60/80 px. The best door result (0.2412) regressed wall IoU
  0.6465 -> 0.6439 and foreground IoU 0.8734 -> 0.8725. The only non-regressing
  allowance changed door by just +0.0006 while reducing window ownership by
  0.0001. These variants were not meaningful and were fully reverted.
- Goal not achieved within the current geometric detector. No source or test
  file from Cycle 9 was retained or committed.

### Cycle 10 goal: distinguish remaining exterior-window candidates

- Audit exact window objects after clean semantic export. Determine whether
  style, shell context, physical wall opening, or source frame evidence can
  eliminate false candidates and recover missing top/left windows without
  reducing either high-quality bottom window.

### Cycle 10 not retained: remaining window cues are not separable

- A goal-matched blue/0.62 palette left window IoU and macro IoU unchanged at
  0.2919 and 0.4927, with only +/-0.0001 ownership noise. It was reverted as a
  cosmetic-only change.
- Individual exact-mask precision is 0.846 and 0.869 for two correct bottom
  windows; two top-shell candidates have zero truth overlap, and a third bottom
  candidate is materially mislocalized (precision 0.014). All survivors already
  satisfy framed-line, exterior room context, shell proximity, tangency, and
  source-face requirements. Prior physical-gap audits also measured identical
  closed-face support for true and false candidates.
- Removing the low-overlap candidates would improve precision on this sheet but
  would neither recover the missing top/left windows nor follow a feature that
  generalizes to other plans. No Cycle 10 source change was retained.

### Final stopping condition after Cycle 10

- The final retained annotation is Cycle 8. Both requested highlighted cases
  remain fixed together: the recreation room is continuous (room IoU 0.9393;
  0 vertical and 2 horizontal residual barrier pixels), and the restored-wall
  crop remains at wall IoU 0.4528 / boundary F1 0.7394 without reintroducing the
  measurement-line wall.
- Whole-image final metrics are wall 0.6465, door 0.2323, window 0.2919, room
  0.7999, macro 0.4927, and foreground 0.8734. The output contains nine stable
  semantic room classes and no diagnostic vectors in the final annotation.
- Meaningful deviations remain in door swing quadrant/localization, the missing
  top/left windows, one mislocalized bottom window, and local wall alignment or
  thickness. Repeated detector, confidence, quadrant, leaf, palette, shell,
  physical-gap, and topology-ownership experiments could not reduce these
  errors without regressing another class or using goal-specific choices.
- Further progress requires multiple labeled plans for a trained semantic/
  instance model (especially door/window examples), or original CAD/PDF layer
  metadata that identifies object semantics. Those resources are unavailable
  in this project, satisfying the requested technical stopping condition.

### Cycle 11 goal: remove duplicate and off-centre door annotations

- A new surgical opening audit exported source/goal/current/difference panels
  for every detected door and window plus every substantial reference component
  in `debug_output/cycle11_opening_audit/`. It found two detections on the same
  guest-suite doorway only about 31 px apart.
- Root cause: Hough confidence did not account for the distance between the
  proposed circle centre and the hinge after snapping to the supporting wall.
  The higher-scored duplicate had a centre-to-hinge offset 1.355 times its
  radius, which is physically impossible for a swing circle. The retained
  detection on that opening has an offset ratio 0.524. Other nearby laundry
  doors are distinct openings and remain under the 0.90 validity limit.
- Change: door candidates now require the Hough centre to remain within 0.90
  leaf radii of the snapped hinge. The check is scale-normalized, tied to door
  geometry, and contains no coordinates or labels from the supplied goal.
- Full rendered results versus Cycle 8: wall IoU 0.6465 -> 0.6476, door IoU
  0.2323 -> 0.2342, window IoU 0.2919 -> 0.2923, room IoU 0.7999 -> 0.8001,
  macro IoU 0.4927 -> 0.4935, and foreground IoU 0.8734 -> 0.8736. Door
  precision improves 0.4961 -> 0.5062; exported doors fall from 8 to 7 and
  door gaps from 13 to 12.
- Both protected structural cases remain exact: measurement crop wall IoU
  0.7169, room IoU 0.9393, 0 vertical / 2 horizontal residual barriers;
  restored-wall crop wall IoU 0.4528, boundary F1 0.7394, room IoU 0.6618.
- Files changed: `config.py`, `door_detection.py`, door unit tests, and this
  log. Commands: per-opening crop export, individual door/window scorer,
  targeted and full CV tests, cached downstream run, full uncached CLI,
  whole-image evaluator, and focused structural evaluator.
- Tests: 172 passed. Full uncached run: 369.8 s, 116 walls, 7 doors, 5 windows,
  9/9 labeled rooms, and 12 gaps. Output:
  `debug_output/highlight_cycle11_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle11_full.json`; masks:
  `debug_output/highlight_cycle11_full_intermediates/13_semantic_masks/`;
  per-opening crops: `debug_output/cycle11_opening_audit/`.

### Cycle 12 goal: recover the mislocalized third bottom window

- The opening audit shows two bottom windows with strong individual precision
  (0.846 and 0.869), while the third detected bottom window is shifted right
  and undersized (precision 0.014). Inspect the source frame-line proposals and
  merge/alignment logic on the same supporting wall. Recover the complete third
  span from repeated frame boundaries without changing the two correct spans,
  top/left wall ownership, room masks, or the Cycle 11 door improvement.

### Cycle 12 retained: recover clipped paired-frame windows

- Root cause: the two source faces of the third bottom window extend only 3 px
  beyond the synthesized supporting wall. The strict `[0, 1]` wall-projection
  test rejected both, even though they have matching 244/249 px spans and are
  the exact paired lines from which wall `W0009` was synthesized.
- A first global 5 px endpoint tolerance was rejected: it manufactured 19
  windows, reducing wall IoU to 0.5998 and macro IoU to 0.4862. A second
  corroborated tolerance still borrowed nearby lines from overlapping wall
  representations and produced 11 windows. Neither experiment was retained.
- Change: small endpoint overruns are now a strict fallback only when no exact
  frame exists. Both corroborating faces must belong to the supporting wall's
  `source_ids`, have distinct offsets, overlap, and agree in length by at least
  0.80. Accepted overrun endpoints are clamped to the wall rather than allowed
  to enlarge it. This recovers one source-supported window without coordinates
  or ground-truth-derived class rules.
- Full rendered results versus Cycle 11: wall IoU 0.6476 -> 0.6569, door IoU
  0.2342 -> 0.2351, window IoU 0.2923 -> 0.3921, room IoU 0.8001 -> 0.8001,
  macro IoU 0.4935 -> 0.5211, and foreground IoU 0.8736 -> 0.8737. Window
  recall improves 0.4148 -> 0.5371 and F1 0.4524 -> 0.5633; detected windows
  increase only from 5 to 6.
- Both protected structural cases remain non-regressing. The measurement crop
  improves from wall IoU 0.7169 to 0.7705 while room IoU stays 0.9393 with
  0 vertical / 2 horizontal residual barrier pixels. The restored-wall crop
  remains wall IoU 0.4528, boundary F1 0.7394, and room IoU 0.6618.
- Files changed: `config.py`, `window_detection.py`, window unit tests, and this
  log. Commands: candidate/source-provenance audit, rejected tolerance sweeps,
  cached downstream comparisons, targeted and full CV tests, full uncached
  CLI, whole-image evaluator, focused structural evaluator, and per-opening
  crop exporter.
- Tests: 174 passed. Full uncached run: 395.6 s, 116 walls, 7 doors, 6 windows,
  9/9 labeled rooms, and 13 gaps. Output:
  `debug_output/highlight_cycle12_full.{pdf,png}`; metrics:
  `evaluation_output/highlight_cycle12_full.json`; masks:
  `debug_output/highlight_cycle12_full_intermediates/13_semantic_masks/`;
  35 focused opening artifacts: `debug_output/cycle11_opening_audit/`.

### Cycle 13 goal: recover the missing left exterior window without duplicates

- The next largest source-supported opening miss is the left exterior window.
  Its wall interval has four mutually consistent, offset frame lines, but the
  current provenance fallback excludes them because paired-wall synthesis
  selected different face IDs. Require stronger repeated-frame evidence than
  the two-line paired fallback, then deduplicate coincident detections across
  overlapping wall representations. Retain only if it adds the missing window
  once and all global and protected-region metrics remain non-regressing.

### Cycle 13 retained: repeated-frame recovery with cross-wall deduplication

- Root cause: the missing left opening has four aligned source frame strokes
  with 166--171 px spans at distinct wall-normal offsets, but paired-wall
  synthesis chose alternate face IDs. Requiring every tolerant stroke to be a
  selected source face was therefore too restrictive for this crowded shell
  junction. Two overlapping wall representations independently described the
  same opening, so accepting the repeated evidence without deduplication would
  export it twice.
- Change: when neither strict containment nor the two-source-face fallback
  produces a frame, a third fallback requires at least three distinct,
  overlapping, length-consistent offsets. Candidates are still limited to a
  5 px endpoint overrun and exterior-context/tangency checks. A separate
  geometry-based pass then collapses coincident, parallel, similar-width
  windows across overlapping walls and keeps the stronger structural wall.
- The left opening is recovered once on `W0042`; one duplicate on `W0054` is
  suppressed. Exterior candidates with only two non-source lines remain
  rejected. No coordinate, room label, or reference-mask feature is used.
- Full rendered results versus Cycle 12: wall IoU 0.6569 -> 0.6633, door IoU
  0.2351 -> 0.2350, window IoU 0.3921 -> 0.4716, room IoU 0.8001 -> 0.8001,
  macro IoU 0.5211 -> 0.5425, and foreground IoU remains 0.8737. Window
  recall improves 0.5371 -> 0.6715, precision 0.5922 -> 0.6130, F1
  0.5633 -> 0.6409, and boundary F1 0.0228 -> 0.1785.
- Both protected structural cases remain exact: measurement crop wall IoU
  0.7705, room IoU 0.9393, 0 vertical / 2 horizontal residual barriers;
  restored-wall crop wall IoU 0.4528, boundary F1 0.7394, room IoU 0.6618.
- Files changed: `config.py`, `window_detection.py`, window unit tests, and this
  log. Commands: tolerant-frame/exterior-context audit, cached downstream run,
  whole-image evaluator, focused evaluator, opening crop exporter, 177-test CV
  suite, and full OCR-enabled uncached CLI. A sandboxed no-OCR run was invalid
  and overwritten; the accepted run used the existing PaddleOCR cache.
- Full uncached run: 369.9 s, 116 walls, 7 doors, 7 windows, 9/9 labeled rooms,
  and 14 gaps. Output: `debug_output/highlight_cycle13_full.{pdf,png}`;
  metrics: `evaluation_output/highlight_cycle13_full.json`; masks:
  `debug_output/highlight_cycle13_full_intermediates/13_semantic_masks/`;
  33 focused opening artifacts: `debug_output/cycle13_opening_audit/`.

### Cycle 14 goal: resolve the remaining top/bottom window mismatches

- The current output still has a shifted short bottom candidate to the right of
  the three reference windows, two unsupported top candidates, and no window
  at the true guest-suite top opening. Audit shell-thickness consistency and
  source-frame intervals for these four cases. First eliminate candidates on
  secondary thin wall representations without affecting the five recovered
  openings, then recover the top span only if independent repeated source
  evidence distinguishes it from the remaining top false positive.

### Cycle 14 retained: require primary exterior-shell support

- Root cause: the shifted bottom candidate is produced on `W0117`, a 30.9 px
  secondary representation of the bottom shell. All five source-supported
  retained openings sit on primary 52--67 px wall pairs. A second top false
  candidate sits on a 37 px representation, but removing both at once caused
  a small room/foreground ownership regression and was rejected.
- Change: the detector now measures the 75th-percentile thickness of
  exterior-context, hull-tangent paired walls for each plan. A window's
  supporting wall must be at least 50% of that plan-relative shell reference.
  This scales with DPI and construction rather than imposing an absolute wall
  width. The 0.60 experiment that removed two candidates was not retained;
  0.50 surgically rejects only the shifted bottom candidate.
- Full rendered results versus Cycle 13: wall IoU 0.6633 -> 0.6694, window IoU
  0.4716 -> 0.4897, room IoU remains 0.8001, macro IoU 0.5425 -> 0.5485,
  and foreground IoU remains 0.8737 (foreground TP improves by 13). Window
  precision improves 0.6130 -> 0.6440 and F1 0.6409 -> 0.6575 without recall
  loss. Door IoU is unchanged within rendered rounding at 0.2350.
- The measurement-line crop improves again: wall IoU 0.7705 -> 0.8033 while
  room IoU stays 0.9393 with 0 vertical / 2 horizontal residual barriers. The
  restored-wall crop remains wall IoU 0.4528, boundary F1 0.7394, and room IoU
  0.6618.
- Files changed: `config.py`, `window_detection.py`, window unit tests, and this
  log. Commands: top-shell frame/classification audit, exterior-thickness
  audit, 0.60/0.50 cached comparisons, targeted and full CV tests, full
  OCR-enabled uncached CLI, whole-image evaluator, focused evaluator, and
  opening crop exporter. Tests: 178 passed.
- Full uncached run: 370.7 s, 116 walls, 7 doors, 6 windows, 9/9 labeled rooms,
  and 13 gaps. Output: `debug_output/highlight_cycle14_full.{pdf,png}`;
  metrics: `evaluation_output/highlight_cycle14_full.json`; masks:
  `debug_output/highlight_cycle14_full_intermediates/13_semantic_masks/`;
  focused artifacts: `debug_output/cycle14_opening_audit/`.

### Cycle 15 goal: prevent rejected windows from suppressing valid doors

- Door detection produces eight candidates, but window conflict resolution
  currently runs before exterior tangency/support filtering. The audit found
  door `D0005` suppressed by a provisional window that is later rejected,
  leaving neither class at that opening. Reorder conflict resolution after all
  window validity and deduplication checks, then evaluate the restored door's
  source support and goal overlap. Retain only if door metrics improve without
  changing the accepted window, wall, or room masks.
