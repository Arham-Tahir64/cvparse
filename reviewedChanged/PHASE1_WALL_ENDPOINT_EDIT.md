# Phase 1 dependency-aware wall endpoint edit log

Date: 2026-07-14

## Goal

Add the first constrained geometry correction that changes the authoritative
wall graph and recomputes its dependents without rerunning CV. A wall endpoint
edit must operate on the shared node so connected walls cannot silently drift
apart.

## Design and behavior

`move_wall_endpoint` is a copy-on-write domain command. Given a model revision,
wall ID, endpoint, and new image coordinate, it:

1. resolves the wall endpoint to its shared graph node;
2. validates finite coordinates and known image bounds;
3. finds every incident wall from the node connectivity graph;
4. rejects an edit that collapses a wall or leaves a hosted opening beyond the
   shortened wall;
5. moves the shared node and updates every incident wall endpoint;
6. recomputes wall length, orientation, boundary polygon, and calibrated
   length;
7. keeps openings at their existing longitudinal offsets and recomputes their
   centers/orientations;
8. rigidly realigns all attached door swing geometry and preserves attached
   windows;
9. moves room vertices anchored to the edited corner and recalculates area,
   perimeter, and calibrated measurements;
10. marks changed objects as manually adjusted, unlocked, and needing review;
11. reruns structural validation and appends one audit event containing the
    inverse-defining before/after node coordinates and every affected ID.

Only graph dependents are changed. No detector or OCR stage is rerun.

The new validation rules also flag:

- wall endpoint coordinates that disagree with their graph nodes;
- node connectivity lists that disagree with wall endpoint references.

## API

Persisted models expose:

```text
PUT /api/takeoff/models/{model_id}/walls/{wall_id}/endpoints/{start|end}
```

The JSON body contains `expected_revision`, `x`, `y`, and optional `actor`.
Stale revisions return HTTP 409. Geometrically invalid edits return HTTP 422.

## Files added or changed

- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/vision/domain/geometry.py`
- `apps/api/src/vision/domain/import_cv.py`
- `apps/api/src/vision/domain/validation.py`
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
  --output-dir evaluation_output\reviewed_changed_wall_endpoint
```

Results:

- Focused domain/API suite: 22 passed.
- Complete suite: 209 passed.
- The five-plan validation summary and every per-class metric are exactly equal
  to `evaluation_output/generalization_cycle3/report.json`.
- Assisted fixture correction: one endpoint action updates exactly eight
  dependent objects (shared node, two walls, two openings, door, window, and
  room); an unrelated graph node is unchanged.
- The command preserves opening containment and graph coordinate consistency.
- An attempted shortening that would orphan an opening is rejected without
  mutating the input model.

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

## Limitations and next slice

- Room corner propagation currently applies when the room polygon has a vertex
  anchored within one quarter of the incident wall thickness from the moved
  node. Imported room-to-wall boundary relationships remain incomplete, so a
  full face reconstruction is not yet available for more distant wall moves.
- Endpoint snapping and node merge/split are not yet implemented.
- Wall add/delete/split/merge and thickness editing are not yet implemented.
- There is still no quantity/cost engine or domain-model renderer, so the
  command updates calibrated geometry but not material assemblies or an
  exported reviewed annotation.
- Undo data is captured in the event, but undo/redo commands do not yet exist.

The next high-impact slice should make rendered annotations consume the
persisted domain model. That closes the current source-of-truth split and makes
this tested human correction visible in exported output before adding more edit
types.
