# Constrained wall creation and deletion

## Iteration goal

Let a reviewer correct the two highest-impact wall candidate failures in the
authoritative takeoff model:

- add a missed wall as connected graph geometry;
- delete a false wall without silently orphaning openings or rooms.

Both operations must immediately affect rendered annotations and quantities,
remain auditable and undoable, and preserve approved-model and optimistic
revision protections.

## Problem before this change

The reviewed model supported scale confirmation, object review, and movement
of an existing shared endpoint. It could not represent a wall that automatic
extraction missed and could not remove a false wall. A reviewer therefore had
no path to correct wall recall or precision errors without rerunning or
manually altering serialized JSON.

## Design

### Add wall

`add_wall` accepts two endpoints, thickness, type, and an optional snap
tolerance. Its default snap distance is adaptive to the proposed wall
thickness: `max(2 px, 0.75 * thickness)`. The command:

1. Validates finite in-page endpoints and positive thickness.
2. Snaps each endpoint to the nearest existing graph node within tolerance.
3. Creates stable manual node IDs only where no reusable node exists.
4. Rejects a collapsed wall, an existing endpoint-pair duplicate, an overlap,
   or a crossing that lacks a shared graph endpoint.
5. Creates a stable manual wall ID, polygon, centerline length, orientation,
   and calibrated length when scale is confirmed.
6. Rebuilds reciprocal node/wall connectivity.
7. Marks reused nodes and structurally adjacent walls as manually adjusted.
8. Marks spatially affected rooms as topology-stale instead of silently
   retaining them as authoritative.
9. Revalidates the model and records one audited revision.

The no-unsplit-crossing rule is intentional. A wall ending in the middle of
another wall must first use a future split-wall operation so the T-junction is
represented by a real node and openings can be reassigned safely.

### Delete wall

`delete_wall` discovers its logical dependencies before modifying a copy:

- host-wall openings;
- doors and windows backed by those openings;
- rooms that explicitly name the wall as a boundary.

The command rejects deletion when dependencies exist unless `cascade=true` is
explicit. Cascade removes the opening/door/window dependency chain, updates
room relations, removes orphan endpoint nodes, rebuilds reciprocal wall
connectivity, invalidates affected room topology, recalculates validation, and
records the exact affected IDs in audit history.

Deleting a newly created wall resolves the room invalidation caused by that
same wall ID. Deleting an original structural boundary creates a new
`room.topology_stale` error. This prevents stale room area from passing final
approval while avoiding a false blocking error when an added false wall is
immediately removed.

### Shared geometry primitives

Reusable point-on-segment, segment-intersection, and inclusive
point-in-polygon operations now live in `vision.domain.geometry`. Split/merge
and room-reconstruction commands can use the same deterministic predicates.

## API additions

- `POST /api/takeoff/models/{model_id}/walls`
- `DELETE /api/takeoff/models/{model_id}/walls/{wall_id}`

Both accept `expected_revision` and `actor`. Create accepts endpoints,
thickness, type, and optional snap tolerance. Delete accepts the explicit
`cascade` decision. Domain constraint failures return HTTP 422 and stale
revisions return HTTP 409.

The new wall ID is available in the final `add_wall` audit event payload. The
entire updated model is returned, matching the existing edit API contract.

## Files modified

- `apps/api/src/vision/domain/geometry.py`
- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/vision/domain/validation.py`
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
  --output-dir evaluation_output\reviewed_changed_wall_add_delete
```

Results:

- Focused domain/API tests: 41 passed.
- Complete test suite: 228 passed, 2 dependency warnings.
- Compile check: passed.
- `git diff --check`: passed.
- Add-wall test: reused a nearby node, created only one new node, updated both
  sides of graph connectivity, added exactly 200 px / 10 ft of wall length,
  invalidated the affected room, and restored the prior graph through undo.
- Constraint test: rejected an existing-wall duplicate, a proper crossing,
  an endpoint placed on an unsplit wall interior, and a partial collinear
  overlap sharing one endpoint.
- Delete-wall test: removed the new wall and orphan node, restored adjacent
  connectivity and wall quantities, resolved its matching room invalidation,
  and restored the deleted state through undo.
- Cascade test: rejected a wall with an opening and door by default, then
  removed the full dependency chain only with explicit cascade.
- API test: persisted create as revision 2, reflected its 90 px length in the
  quantities endpoint, persisted delete as revision 3, and restored the wall
  with one undo as revision 4.

## Automatic first-pass regression result

Automatic extraction is unchanged. Every class result for all five validation
plans exactly matches the prior undo/redo report:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

The generated report remains intentionally untracked at
`evaluation_output/reviewed_changed_wall_add_delete/report.json`.

## Assisted-workflow improvement

A missed standalone or corner-connected wall now requires one create action;
a dependency-free false wall requires one delete action. The automatic API
test verifies that quantities change in the same persisted revision. A
dependent wall takes two deliberate user decisions: inspect the reported IDs,
then repeat with cascade if deletion is intended. Undo is one action in all
cases.

Real-user time and click measurements remain unavailable until the review UI
is implemented. The automated action counts are behavioral evidence, not a
substitute for usability measurement.

## Remaining limitations

- A T-junction or crossing wall is rejected until split-wall support exists.
- Rooms are conservatively marked stale; the wall command does not yet rebuild
  room faces from the edited graph.
- There is no merge-wall command or wall-thickness edit command.
- Cascade is all-or-nothing. Reassigning openings to another wall before
  deletion requires opening edit commands that are not implemented yet.
- Dependency detection is strongest for explicit IDs. Imported rooms do not
  yet have complete `boundary_wall_ids`, so spatial adjacency is used to
  invalidate them but cannot provide a full formal dependency list.
- No graphical editor currently invokes these endpoints.

## Next highest-impact slice

Implement wall splitting at a constrained projected point. Preserve the
original wall ID on one segment, create one new segment and shared node,
reassign openings by offset, update room boundary relations, rebuild graph
connectivity, recalculate quantities, and prove undo/redo. This will make
T-junction wall creation usable without compromising topology.
