# Opening-preserving wall split

## Iteration goal

Add a safe wall split operation so reviewers can create T-junctions without
corrupting graph topology, changing total wall quantities, losing stable IDs,
or detaching openings.

## Problem before this change

Constrained wall creation correctly rejected any endpoint placed in the
interior of an unsplit wall. That protected graph invariants, but it also meant
a common missing T-junction could not be corrected. There was no command to
insert the required shared node and divide the host wall while preserving its
opening relationships.

## Design

`split_wall` accepts a wall ID and a reviewer-selected point. The point is
projected onto the existing centerline, with a default adaptive acceptance
distance of `max(2 px, wall thickness)`. The command:

1. Rejects non-finite, off-page, or excessively distant clicks.
2. Requires both resulting segments to exceed an endpoint margin based on wall
   thickness.
3. Rejects a split through the interior of any opening, naming the opening IDs
   that must first be moved or resized.
4. Reuses an existing on-centerline interior node when safe; otherwise creates
   one stable manual node ID at the projection.
5. Preserves the original wall ID on the start segment.
6. Creates one stable manual wall ID for the end segment and records its parent
   wall ID as provenance.
7. Recalculates both polygons, orientations, pixel lengths, and calibrated
   lengths.
8. Keeps openings before the split on the parent. Openings after the split move
   to the child with offsets translated to the child's local coordinate system.
9. Marks reassigned openings and their door/window dependents as manually
   adjusted while preserving their logical object IDs and world geometry.
10. Rebuilds reciprocal node and wall connectivity.
11. Expands explicit room boundary relations from the parent ID to the ordered
    parent/child pair without changing room geometry or confirmed status.
12. Preserves existing room-topology invalidation across both child segments
    when a room was already stale.
13. Produces one new audited model revision with complete affected IDs and
    projection/reassignment details.

The sum of the two segment lengths equals the original wall length, so wall
takeoff quantities do not drift. The structured model remains the source for
annotations and quantities, and the operation participates in the existing
undo/redo revision mechanism.

## Shared geometry addition

`vision.domain.geometry.project_point_to_segment` returns the clamped projected
coordinate, longitudinal wall offset, and lateral click distance. It is shared
domain geometry for split, future constrained opening movement, and other
wall-local edits.

## API addition

`POST /api/takeoff/models/{model_id}/walls/{wall_id}/split`

Request fields:

- `expected_revision`
- `x`, `y`
- optional `projection_tolerance_px`
- optional `actor`

The updated model is returned. The new wall ID, split node ID, projected point,
offset, reassigned opening IDs, and affected room IDs are available in the
final `split_wall` audit event. Constraint failures return HTTP 422; stale
revisions return HTTP 409.

## Files modified

- `apps/api/src/vision/domain/geometry.py`
- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/api/routes/takeoff_models.py`
- `tests/unit/vision/domain/test_domain_model.py`
- `tests/unit/vision/cv/test_api_route.py`

## Verification

Commands run from the repository root:

```powershell
$env:PYTHONPATH='apps/api/src'
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\vision\domain\test_domain_model.py `
  tests\unit\vision\cv\test_api_route.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_wall_split
```

Results:

- Focused domain/API tests: 45 passed.
- Complete test suite: 232 passed, 2 dependency warnings.
- Compile check: passed.
- `git diff --check`: passed.
- Projection/reassignment test: a click 2 px off the centerline projected to
  `(220, 80)`; the stable 200 px parent became 60 px + 140 px; a window opening
  moved from offsets 75-105 to child offsets 15-45; world center, total wall
  length, total opening width, floor area, and room perimeter stayed exact.
- Room relationship test: `[parent]` became `[parent, child]` while a confirmed,
  locked room remained confirmed and locked because its physical boundary did
  not change.
- Invalid split test: rejected a through-opening point, an off-wall click, and
  a point at the wall endpoint without mutating revision state.
- T-junction test: split plus add created one shared node with degree three and
  added exactly 100 px of new wall length.
- History test: undo restored the original single wall and opening host; redo
  restored the split and child stable ID.
- API test: a persisted 90 px manual wall became two 45 px segments at revision
  3, quantities remained current at revision 3, and one undo restored the 90 px
  parent with no child.

## Automatic first-pass regression result

This reviewed-domain operation does not change automatic extraction. Every
class result for all five plans exactly matches the preceding wall-correction
baseline:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

The generated report is intentionally untracked at
`evaluation_output/reviewed_changed_wall_split/report.json`.

## Assisted-workflow improvement

A missing T-junction that was previously impossible to represent now takes two
domain actions: split the host wall, then add the missing branch. The automated
test proves both actions preserve graph validity and quantities. Each action is
independently undoable. A split with a downstream opening performs the host
reassignment in the same action rather than requiring a delete/recreate cycle.

Real-user duration and click count remain to be measured in the future review
interface.

## Remaining limitations

- A split cannot pass through an opening; the reviewer must move or resize that
  opening first.
- There is no merge-wall operation to reverse a split semantically in one edit;
  undo works only while the split remains in history.
- The split operation preserves existing room polygons. Adding the branch then
  marks affected rooms stale; graph-to-room face recomputation is still needed.
- Imported rooms still lack complete boundary wall IDs on many plans.
- There is no graphical gesture layer for choosing the projected split point.
- Multi-process persistence still requires a transactional database repository.

## Next highest-impact slice

Implement constrained opening correction: add an opening to a wall, move it
along the wall axis, resize it within host bounds, change its class between
door/window/archway/unknown, and delete it with explicit logical dependency
handling. This directly addresses the weakest automatic door/window metrics
while keeping corrections fast and quantity-aware.
