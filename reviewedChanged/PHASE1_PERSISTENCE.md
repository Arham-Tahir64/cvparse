# Phase 1 persistence and review log

Date: 2026-07-14

## Goal

Make the editable takeoff model durable and give reviewers the first safe,
revision-checked human actions. Preserve the legacy CV response and all detector
metrics.

## Implemented

- Added in-memory and JSON-file model repositories behind a shared repository
  protocol.
- JSON writes use a temporary file plus atomic replacement. Model IDs are
  restricted to safe filename characters.
- Every update uses optimistic concurrency: clients must supply the revision
  they read, incoming revisions must advance by exactly one, and stale updates
  return HTTP 409 instead of overwriting newer work.
- Added copy-on-write commands for:
  - confirming drawing scale;
  - recalculating wall lengths, opening widths, room areas, and room perimeters;
  - confirming, rejecting, or reopening any node, wall, opening, door, window,
    or room;
  - locking confirmed/rejected objects by default so later automatic work can
    distinguish reviewed geometry.
- Every successful command records an immutable edit event with actor,
  timestamp, before/after revision, affected IDs, and before/after values.
- Validation is recomputed after each command without rerunning CV.
- Added API operations to persist a CV import, retrieve it, set scale, and
  change one object's review state.

## API workflow

1. `POST /api/cv/takeoff` with multipart `persist_model=true` creates revision
   1 and includes the model in the response.
2. `GET /api/takeoff/models/{model_id}` retrieves the current revision.
3. `PUT /api/takeoff/models/{model_id}/scale` sets a manual scale using
   `expected_revision`, `pixels_per_unit`, and `unit`.
4. `PUT /api/takeoff/models/{model_id}/objects/{object_id}/review` changes the
   review status using `expected_revision` and `status`.

The default repository writes to `data/models`; `FLOWBUILDR_MODEL_DIR` can
select a different location. The file repository is intended for the current
single-process service. A transactional database repository is required before
running multiple API worker processes.

## Files added or changed

- `apps/api/src/api/main.py`
- `apps/api/src/api/model_store.py`
- `apps/api/src/api/routes/cv_takeoff.py`
- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/vision/domain/models.py`
- `apps/api/src/vision/domain/repository.py`
- `apps/api/src/vision/domain/serialize.py`
- `tests/unit/vision/cv/test_api_route.py`
- `tests/unit/vision/domain/test_domain_model.py`

## Verification

Commands:

```powershell
$env:PYTHONPATH = "apps/api/src"
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_phase1_persistence
```

Results:

- Focused domain/API suite: 16 passed.
- Complete suite: 203 passed.
- Persist/get/scale/review/stale-write/duplicate-import API workflow passes.
- JSON reload and edit-history round trip are lossless.
- The five-plan validation summary and every per-class metric are exactly equal
  to `evaluation_output/generalization_cycle3/report.json`.

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

The first sandboxed validation attempt could not read the existing PaddleOCR
cache outside the workspace, so OCR initialization failed and the harness did
not reproduce its baseline. Re-running with read access to the same cached
models used by the baseline completed successfully. Generated evaluation files
remain untracked.

## Remaining limitations and next slice

- There are not yet geometry edit commands for moving/adding/deleting walls or
  openings.
- Rooms do not yet carry explicit wall/opening boundary relationships.
- There is no undo/redo command, model approval/reopen workflow, or takeoff
  quantity/cost engine.
- The annotation renderer still consumes legacy CV output rather than the
  reviewed model, so human corrections cannot yet drive a regenerated overlay.
- There is no review UI in this backend-only repository.

The next slice should implement dependency-aware geometry edits and
recalculation, beginning with wall endpoint edits that update connected walls,
hosted openings, adjacent room validity, quantities, and validation issues
without rerunning unrelated detectors.
