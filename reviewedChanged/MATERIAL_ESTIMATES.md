# Revision-bound material estimates

## Iteration goal

Close the largest gap between an editable geometry model and a usable
construction takeoff: derive material quantities, waste-adjusted purchase
quantities, and costs from the same reviewed model revision without inventing
scale, dimensions, rates, or authoritative status.

## Why the prior architecture was incomplete

`calculate_quantities` already provided wall length, opening width, floor and
ceiling area, room perimeter, and object counts with correct provisional versus
verified inclusion semantics. It intentionally stopped before wall height,
opening deductions, assemblies, waste, and rates. Consequently a corrected
model could render and report geometry but could not produce a material or cost
estimate, and there was no proof that a geometry correction propagated into
estimating results.

## Design

`vision.domain.costs.calculate_material_estimate` is a pure derived service. It
does not create a second geometric source of truth. It first obtains the
included object set from `calculate_quantities` for the requested review basis,
then derives every line from those exact object IDs and the current model
revision.

The caller supplies all project assumptions:

- wall, door, and window height;
- wall-finish and insulation sides;
- stud spacing, plate count, and extra opening studs;
- opening-trim sides;
- per-material waste factors;
- per-material unit rates; and
- currency.

There are no plan coordinates, fixture dimensions, vendor prices, or hidden
default costs in production code. The only defaults are conventional formula
shape (for example, two wall-finish sides and three plates), and every default
is exposed in the request.

Each result line records:

- raw quantity and calibrated unit;
- waste factor and purchase quantity;
- optional unit and extended cost;
- source wall, room, opening, door, or window IDs; and
- the model ID, model revision, and provisional/verified basis on the summary.

Implemented lines are drywall, paint, insulation, framing lumber, flooring,
ceiling, baseboard, door units, window units, glazing, door trim, and window
trim. Wall finishes and insulation deduct included door/window openings.
Baseboard deducts a door once per known adjacent room. If imported room-door
relationships are missing, it deducts once, emits an explicit warning, and
prevents the estimate from becoming authoritative until topology is repaired.

Unconfirmed scale produces `null` physical quantities rather than pixel values
misrepresented as construction units. Missing positive-quantity rates produce
a partial priced subtotal plus named warnings. `authoritative=true` requires a
complete verified geometry chain, complete pricing, and resolved door-room
associations.

## API

`POST /api/takeoff/models/{model_id}/estimates/materials`

The request includes `expected_revision`; a stale estimate request returns HTTP
409. Invalid material codes or assumptions return HTTP 422. This is read-only:
it derives from, but does not mutate, the authoritative model revision.

Example request fields:

```json
{
  "expected_revision": 12,
  "basis": "verified",
  "wall_height": 8,
  "door_height": 7,
  "window_height": 4,
  "stud_spacing": 1.333333,
  "waste_factors": {"flooring": 0.1, "drywall": 0.12},
  "unit_costs": {"flooring": 3.5, "drywall": 0.85},
  "currency": "CAD"
}
```

## Files modified

- `apps/api/src/vision/domain/costs.py`
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
  --output-dir evaluation_output\reviewed_changed_material_estimates
```

Results:

- Focused domain/API tests: 61 passed.
- Complete suite: 248 passed, with two dependency warnings.
- Exact 20 ft wall / 100 ft² room fixture at 8 ft wall height:
  - 280 ft² net drywall and paint after a 2×7 ft door and 1.5×4 ft window;
  - 140 ft² single-cavity-side insulation;
  - 188 linear ft stud-and-plate material at 2 ft spacing;
  - 100 ft² flooring, purchased as 110 ft² with 10% waste;
  - 38 linear ft baseboard, 6 ft² glazing, 32 ft door trim, and 22 ft
    window trim; and
  - exact priced subtotal of 2,021 for the test catalog.
- Resizing the door from 2 ft to 3 ft advanced the model revision, reduced net
  drywall by 14 ft² and baseboard by 1 ft, and left ceiling area unchanged.
- An unscaled plan returned no physical material quantities or false costs.
- A missing door-room relationship emitted a warning and prevented an otherwise
  verified estimate from becoming authoritative.
- The persisted API rejected a stale revision and an unknown material code.

## Automatic first-pass regression result

This derived service does not alter extraction. All 20 class/case confusion
matrices exactly match the prior benchmark:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Generated evidence remains untracked at
`evaluation_output/reviewed_changed_material_estimates/report.json`.

## Remaining limitations

Assumptions and catalogs are request inputs and are not yet persisted as
revisioned project configuration. A future slice should add reusable named
assemblies and vendor catalogs without coupling them to plan geometry. Wall
height and opening heights are project-wide because the current object model
does not yet carry per-wall or per-opening height. Interior/exterior wall types,
headers, corner studs, cripples, sheet/package rounding, labor, taxes, markup,
and multi-currency conversion are not modeled. Cost-impact review priorities
are not yet recalculated from this estimate. There is also no estimate-review UI
or assisted-completion timing instrumentation yet.
