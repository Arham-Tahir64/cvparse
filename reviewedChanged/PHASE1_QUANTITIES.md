# Phase 1 model-native quantity log

Date: 2026-07-14

## Goal

Calculate the first structured takeoff totals from the same persisted model and
revision used by reviewed annotation output. Automatic candidates must never be
silently presented as final verified quantities.

## Inclusion model

The quantity engine exposes two explicit bases:

- `provisional`: includes every non-rejected candidate with a valid dependency
  chain. It always reports `authoritative: false` and warns that automatic
  candidates are included.
- `verified`: includes only confirmed objects whose required host relationships
  are also confirmed. It is authoritative only when scale is confirmed, every
  non-rejected quantity object is included, and no included/model-wide
  structural error remains.

Dependency examples:

- an opening requires its included host wall;
- a door or window requires its included opening, which in turn requires its
  included wall;
- rejecting or leaving any dependency unconfirmed prevents that logical object
  from entering verified totals;
- an excluded active object makes the verified result partial rather than
  allowing a misleading completeness claim.

## Implemented quantities

Both pixel and, when scale is confirmed, calibrated totals are returned for:

- wall centerline length;
- total opening width;
- room/floor area;
- ceiling area (currently equal to floor area);
- room perimeter;
- wall, opening, door, window, and room counts.

Every result includes model ID, model revision, basis, unit, scale status,
completeness/authority flags, included and excluded stable IDs, and explanatory
warnings.

Endpoint:

```text
GET /api/takeoff/models/{model_id}/quantities?basis=provisional|verified
```

## Files added or changed

- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/domain/quantities.py`
- `tests/unit/vision/cv/test_api_route.py`
- `tests/unit/vision/domain/test_domain_model.py`

## Verification

Commands:

```powershell
$env:PYTHONPATH = "apps/api/src"
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\vision\domain\test_domain_model.py `
  tests\unit\vision\cv\test_api_route.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_quantities
```

Results:

- Focused domain/API suite: 27 passed.
- Complete suite: 214 passed.
- Fully confirmed fixture at 20 px/ft: 20 ft wall centerline, 100 ft2
  floor/ceiling area, 3.5 ft total opening width, one door, and one window.
- Unscaled provisional fixture: all physical totals are null, while pixel
  totals and counts remain available and explicitly non-authoritative.
- A confirmed door with an unconfirmed opening contributes zero verified doors.
- The persisted API workflow proves one wall endpoint edit changes the
  revision-2 wall-length quantity and the overlay at the same revision.
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

- Costs, waste factors, wall height/surface area, paint, drywall, insulation,
  framing, flooring assemblies, and material catalogs are not implemented.
- Ceiling area assumes a flat ceiling matching floor area.
- Wall surface quantities require explicit wall height and interior/exterior
  face semantics.
- Room boundary relationships remain incomplete for imported plans.
- The summary is derived on request rather than persisted as a cache; this is
  intentional until dependency tracking covers all edit types.

The next quantity slice should introduce versioned assemblies and user-supplied
wall heights, waste factors, unit costs, and cost-impact validation. Source
asset persistence and combined reviewed-PDF export remain the other major
Phase 1 gap.
