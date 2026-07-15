# Phase 1 implementation log

Date: 2026-07-14

## Goal

Create the smallest usable boundary between automatic CV proposals and a
future authoritative, editable takeoff model. Do not change detector behavior
or the existing schema 1.0.0 response.

## Implemented

- Added `vision.domain`, containing:
  - a versioned `TakeoffModel` (`2.0.0-alpha.1`);
  - plan source fingerprint and scale calibration records;
  - stable nodes, walls, openings, doors, windows, and rooms;
  - object source/provenance, confidence breakdown, review status, lock, and
    revision metadata;
  - validation issues with uncertainty, structural impact, cost impact, and
    deterministic priority;
  - deterministic import from `CVTakeoffResult`;
  - lossless JSON serialization and rehydration.
- The importer reconstructs shared wall nodes and connectivity.
- Gaps become authoritative `Opening` objects. Door and window records reference
  the same physical opening instead of independently encoding wall cuts.
- Imported automatic objects are never marked `confirmed`.
- Detector IDs, stages, wall evidence, and relevant gap metrics remain attached
  as source evidence.
- Structural validation currently flags:
  - unconfirmed drawing scale;
  - dangling wall nodes;
  - invalid wall geometry;
  - missing opening host walls;
  - invalid/out-of-wall opening ranges;
  - substantially overlapping openings;
  - doors/windows with invalid or duplicate opening references;
  - invalid room polygons.
- Added backward-compatible `include_model=true` to `POST /api/cv/takeoff`.
  The upload SHA-256 supplies the stable source fingerprint. When the option is
  absent, the legacy response shape is unchanged.

## Files added or changed

- `reviewedChanged/ARCHITECTURE_ASSESSMENT.md`
- `reviewedChanged/PHASE1_IMPLEMENTATION.md`
- `apps/api/src/vision/domain/__init__.py`
- `apps/api/src/vision/domain/models.py`
- `apps/api/src/vision/domain/import_cv.py`
- `apps/api/src/vision/domain/serialize.py`
- `apps/api/src/vision/domain/validation.py`
- `apps/api/src/api/routes/cv_takeoff.py`
- `tests/unit/vision/domain/test_domain_model.py`
- `tests/unit/vision/cv/test_api_route.py`

## Verification

Commands:

```powershell
$env:PYTHONPATH = "apps/api/src"
.\.venv\Scripts\python.exe -m compileall -q apps\api\src\vision\domain
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_phase1
```

Results:

- Focused domain/API tests: 12 passed.
- Complete suite: 199 passed in 6.45 seconds.
- Model JSON round trip is exact.
- Legacy schema remains `1.0.0` and contains no new review/model fields.
- Existing API responses omit `model` unless explicitly requested.
- All five automatic CV class-metric dictionaries and the summary are exactly
  equal to `evaluation_output/generalization_cycle3/report.json`.

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Worst floorplan: `degraded_dense` at macro IoU 0.3531. Worst class/case:
`compact_thin` windows at IoU 0.0000.

## Limitations and next slice

This is an importable/editable schema boundary, not yet a complete review
system. There is no persistence repository, edit command service, undo/redo,
confirmed-object precedence, scale-setting endpoint, quantity engine, or UI.
Room boundary relationships are not inferred during import yet. Domain
rendering still uses the legacy result/mask path.

The next Phase 1 slice should add a model repository plus revision-checked
scale and review-state commands. That provides the first persisted human action,
establishes manual-over-automatic precedence, and allows calibrated quantities
to be introduced without rerunning CV.

