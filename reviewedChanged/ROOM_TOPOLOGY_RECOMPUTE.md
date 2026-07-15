# Reviewed wall graph to room topology

## Iteration goal

Replace conservative stale-room warnings with an explicit, audited operation
that reconstructs room faces from the current reviewed wall graph after wall
corrections. The operation must preserve stable room identity where geometry
still corresponds, create or remove rooms for real face splits and merges, and
make the reconstructed model the shared source for annotations and quantities.

## Why the prior behavior was incomplete

Wall edits updated graph connectivity and marked affected room polygons stale,
but they deliberately left imported raster room polygons unchanged. This was a
safe warning mechanism, not edit propagation: closing an incomplete boundary,
adding a divider, or deleting a divider could not produce authoritative room
geometry, boundary relations, adjacency, or corrected floor quantities.

## Design

`vision.domain.topology.recompute_room_faces` performs deterministic,
non-machine-learning centerline polygonization:

1. Ignore walls explicitly rejected during review.
2. node the active wall centerlines with Shapely's unary union;
3. polygonize every closed face and reject faces smaller than an adaptive
   wall-thickness-squared floor;
4. match new faces to prior rooms by polygon IoU (minimum 0.25) so surviving
   rooms keep stable IDs and labels;
5. create IDs only for unmatched faces and report rooms that disappeared;
6. derive boundary wall, door, window, and neighboring-room relationships from
   the reconstructed faces;
7. recalculate pixel and calibrated area/perimeter; and
8. mark results for review, clear stale-topology provenance only after a real
   recomputation, then rerun structural validation.

The command is explicit rather than automatic after every wall gesture. This
keeps the wall edit small and auditable while allowing a client to batch several
related corrections before rebuilding rooms. The command creates one immutable
revision and participates in the existing undo/redo snapshot history.

## API

`POST /api/takeoff/models/{model_id}/rooms/recompute`

The request uses the existing history payload with `expected_revision` and
optional `actor`. Optimistic conflicts remain HTTP 409, invalid or open graphs
are HTTP 422, and approved models remain locked. The response is the complete
new authoritative model.

## Files modified

- `apps/api/src/vision/domain/topology.py`
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
  --output-dir evaluation_output\reviewed_changed_room_topology
```

Results:

- Focused domain/API tests: 58 passed.
- Complete suite: 245 passed, with two dependency warnings.
- A corrected four-wall graph reconstructed one 40,000 px² room, preserved its
  original ID and `OFFICE` label, attached all four walls and the existing door
  and window, cleared the stale issue, and updated the floor-area quantity.
- Adding a graph divider reconstructed two 20,000 px² rooms with reciprocal
  adjacency, preserved one existing ID, and created exactly one new ID.
- Deleting that divider and recomputing merged the faces back to one 40,000 px²
  room, removed exactly one room ID, and removed the obsolete adjacency.
- Undo restored the stale pre-recompute geometry; redo restored the same stable
  reconstructed room ID.
- The persisted API workflow added four walls over revisions 2–5, reconstructed
  their 8,100 px² face at revision 6, exposed it through quantities, and removed
  it again through one undo.

## Automatic first-pass regression result

This is a reviewed-model operation and does not alter automatic extraction.
All 20 class/case confusion-matrix results exactly match the prior benchmark:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Generated benchmark evidence remains untracked at
`evaluation_output/reviewed_changed_room_topology/report.json`.

## Assisted-workflow result and limitations

Closing a room, splitting a room, or merging two rooms now requires the wall
correction actions plus one room-recompute action; the recomputation itself is
one auditable action and one undo. Corrected geometry, annotations, validation,
and quantities use the same model revision.

The current face geometry follows wall centerlines, not inner wall finish faces,
so finish-area deductions and exact net room dimensions still require a later
wall-boundary offset model. Polygon holes and multi-level/overlapping plans need
explicit policy. An unmatched manually adjusted room can be removed by this
explicit command, so the future review UI must preview its created/matched/
removed set before applying. Direct room rename, boundary adjustment, manual
create/delete, and label-aware split/merge tools remain to be implemented.
