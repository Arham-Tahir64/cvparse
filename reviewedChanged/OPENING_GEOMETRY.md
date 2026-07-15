# Constrained opening creation and geometry editing

## Iteration goal

Give reviewers a fast, structured way to add a missed door/window opening and
correct an existing opening's localization and width without detaching it from
its host wall or desynchronizing door annotation geometry from quantities.

## Problem before this change

Openings could only come from automatic detection. Reviewers could not add a
missed opening, move an off-center door/window, or correct its width. This was
especially limiting because doors and windows remain the weakest automatic
classes on several validation plans.

## Design

### Create

`add_opening` takes a host wall, plan click, width, semantic kind, and optional
projection tolerance. It projects the click onto the wall centerline, uses an
adaptive default tolerance of `max(2 px, wall thickness)`, requires the whole
opening range to stay inside the wall, and rejects overlap with any existing
opening on that wall.

The command creates one stable manual opening ID. Door and window kinds also
create one stable logical `Door` or `Window` ID; archway/unknown remain logical
openings without a fabricated symbol. Wall and explicit room relationships,
calibrated width, provenance, validation, quantities, audit history, and
undo/redo all update in the same revision.

### Move and resize

`update_opening_geometry` reprojects a new center to the current host wall,
checks wall bounds and overlap, updates wall-local offsets, world center,
pixel/calibrated width, and marks the opening and dependencies as manually
adjusted.

For doors, swing geometry moves and resizes with the opening. The hinge is
translated by the center delta; the leaf endpoint and every arc point are
scaled radially around the new hinge by `new_width / old_width`. Thus the
rendered door leaf radius and the takeoff opening width cannot retain different
sizes after a correction.

## API additions

- `POST /api/takeoff/models/{model_id}/walls/{wall_id}/openings`
- `PUT /api/takeoff/models/{model_id}/openings/{opening_id}/geometry`

Both require `expected_revision` and accept `actor`. Create accepts `x`, `y`,
`width_px`, `kind`, and optional projection tolerance. Geometry update accepts
the new center, width, and optional tolerance. Constraint failures return HTTP
422; stale revisions return HTTP 409.

## Files modified

- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/api/routes/takeoff_models.py`
- `tests/unit/vision/domain/test_domain_model.py`
- `tests/unit/vision/cv/test_api_route.py`

## Verification

```powershell
$env:PYTHONPATH='apps/api/src'
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\vision\domain\test_domain_model.py `
  tests\unit\vision\cv\test_api_route.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_opening_geometry
```

Results:

- Focused domain/API tests: 50 passed.
- Complete suite: 237 passed, 2 dependency warnings.
- Compile and `git diff --check`: passed.
- Create test: a click 3 px off-wall projected exactly to the centerline,
  created one 20 px / 1 ft door opening and logical door, updated wall/room
  relationships and quantities, rendered from the same model, and undid both
  objects in one action.
- Constraint tests: rejected overlap, off-wall placement, host-bound overflow,
  a no-op edit, and movement into another opening.
- Geometry test: moved a 40 px door by 50 px and resized it to 30 px; opening
  width decreased by exactly 10 px, hinge translated 50 px, and leaf/arc radius
  scaled from 40 px to 30 px; undo restored all original geometry.
- API workflow: persisted wall creation at revision 2, door creation at 3,
  move/resize at 4, verified updated quantities and annotation center/width,
  then restored the original opening geometry with one undo.

## Automatic first-pass regression result

All 20 class/case results exactly match the prior split-wall benchmark:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Generated evidence remains untracked at
`evaluation_output/reviewed_changed_opening_geometry/report.json`.

## Assisted-workflow improvement

A missed door or window now takes one create action. An off-center or
incorrect-width opening takes one geometry action, including dependent swing
geometry and quantity recalculation. Each is independently undoable. Actual
human duration and click counts remain to be collected in the review UI.

## Remaining limitations and next slice

- Manual doors are created without invented hinge/swing evidence; subtype,
  hinge side, and swing editing are still needed.
- Opening kind cannot yet be changed after creation.
- Openings and logical door/window objects cannot yet be deleted independently.
- Reassigning an opening to another wall is not implemented.
- Wall finish, framing, and cost deductions are not yet part of quantities.

Next implement opening reclassification and deletion with explicit logical
dependency handling, followed by door subtype/swing correction and room-face
recomputation after structural edits.
