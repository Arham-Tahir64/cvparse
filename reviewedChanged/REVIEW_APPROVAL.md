# Review approval and reopen log

Date: 2026-07-15

## Goal

Make model approval an enforced state transition instead of a decorative field.
Only a complete verified takeoff may be frozen, and no edit may change that
revision until a reviewer explicitly reopens it.

## Implemented

- Added the revisioned `set_approval_status` domain command.
- Approval to `approved` recomputes validation and verified quantities, then
  requires:
  - confirmed drawing scale;
  - every non-rejected quantity object included through a confirmed dependency
    chain;
  - no blocking structural validation error;
  - `complete: true` and `authoritative: true` verified totals.
- Failed approval is copy-on-write: the original model, revision, and audit
  history remain unchanged.
- Approved models reject scale, review-state, and geometry edits through the
  shared command precondition.
- An approved model may transition only to `in_review`. This explicit reopen is
  a new model revision and timestamped audit event.
- Once reopened, normal commands work again and produce later revisions.
- Added:

```text
PUT /api/takeoff/models/{model_id}/approval
```

The JSON body contains `expected_revision`, `status`, and optional `actor`.
Stale writes return 409; invalid or incomplete transitions return 422.

## Files changed

- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/domain/commands.py`
- `tests/unit/vision/cv/test_api_route.py`
- `tests/unit/vision/domain/test_domain_model.py`

## Verification

Commands:

```powershell
$env:PYTHONPATH = "apps/api/src"
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_approval
```

Results:

- Focused domain/API suite: 31 passed.
- Complete suite: 218 passed.
- An unscaled automatic model cannot be approved and remains unchanged at
  revision 1.
- A scaled fixture with every wall/opening/room/door/window confirmed becomes
  approved, retains authoritative verified quantities, and records the actor.
- A scale edit against the approved model is rejected.
- Explicit reopen advances the revision to `in_review`; the same scale edit
  then succeeds as a later audited revision.
- The API rejects approval of a partially reviewed persisted model with a clear
  422 response.
- The five-plan validation summary and every per-class metric are exactly equal
  to `evaluation_output/generalization_cycle3/report.json`.

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

## Remaining limitations and next slice

- Approval currently requires direct confirmation/rejection of all quantity
  objects; bulk review operations and issue acknowledgement are not available.
- Approval does not create a signed immutable external artifact or reviewer
  signature. It freezes the model through domain command enforcement.
- Undo/redo is not implemented. Reopen permits new edits but cannot yet restore
  a prior geometry revision.
- There is no explicit `reset to automatic` transition for manual objects.

The next slice should persist every model revision and add undo/redo stacks.
Undo must create a new audited revision representing an older snapshot; redo
must restore the abandoned snapshot; a new edit after undo must clear redo.
