# Opening reclassification and deletion

## Iteration goal

Complete the physical opening lifecycle needed to correct false door/window
classes and duplicate detections without replacing the authoritative opening
geometry or silently deleting logical dependencies.

## Design

### Reclassification

`set_opening_kind` preserves the stable physical `Opening.id`, wall host,
offsets, center, and width. It removes any logical objects attached to the old
class and creates exactly one new logical `Door` or `Window` when required.
Archway and unknown classifications intentionally have no fabricated symbol
object. Room door/window relationships migrate to the new logical ID, review
metadata becomes manually adjusted, quantities and annotations use the same
new revision, and the event records every removed and created ID.

### Deletion

`delete_opening` discovers dependent `Door` and `Window` objects first. The
default request rejects deletion and reports their IDs. Only explicit
`cascade=true` removes the physical opening and logical dependency chain,
updates its host wall and room relationships, recalculates quantities and
validation, and writes an audited revision. An unclassified opening with no
logical dependent can be deleted without cascade.

Both commands honor approved-model locks, optimistic revisions, immutable
history snapshots, undo/redo, and manual-authority precedence.

## API additions

- `PUT /api/takeoff/models/{model_id}/openings/{opening_id}/kind`
- `DELETE /api/takeoff/models/{model_id}/openings/{opening_id}`

Kind update accepts `expected_revision`, `kind`, and `actor`. Delete accepts
`expected_revision`, `cascade`, and `actor`. Constraint failures return HTTP
422; stale clients return HTTP 409.

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
  --output-dir evaluation_output\reviewed_changed_opening_lifecycle
```

Results:

- Focused domain/API tests: 55 passed.
- Complete suite: 242 passed, 2 dependency warnings.
- Compile and `git diff --check`: passed.
- Door-to-window test: retained the physical opening ID and width, removed the
  old door ID, created one window ID, migrated room relations, changed counts
  by door -1/window +1, preserved total opening width, and rendered only the
  new logical object.
- Undo/redo test: undo restored the exact original door ID and classification;
  redo restored the exact new window ID.
- Delete test: default deletion reported the dependent ID and made no change;
  explicit cascade removed the opening/door chain, host-wall relation, room
  relation, count, and exact opening width; undo restored the chain.
- API workflow: create at revision 3, reclassify at 4, guarded delete returned
  422, cascade delete persisted revision 5, annotations/quantities excluded the
  objects, and one undo restored the window and physical opening at revision 6.

## Automatic first-pass regression result

All 20 class/case results exactly match the previous benchmark:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Generated evidence is intentionally untracked at
`evaluation_output/reviewed_changed_opening_lifecycle/report.json`.

## Assisted-workflow result and limitations

A wrong door/window class now takes one action. A dependent duplicate or false
opening takes one guarded attempt plus one explicit cascade decision; a known
intent can delete directly in one cascade action. Undo is one action.

Remaining opening work includes wall reassignment and door subtype, hinge side,
and swing-direction editing. The higher structural priority now is graph-driven
room-face recomputation after wall add/delete/T-junction edits, which will
replace conservative stale-room flags with recalculated polygons and adjacency.
