# Audited undo and redo

## Iteration goal

Make human corrections reversible without mutating or deleting prior model
revisions. A restored state must remain a new revision, retain append-only edit
history, respect optimistic concurrency, survive a service restart, and never
bypass the approved-model freeze.

## Problem before this change

Edit events recorded what a command changed, but they were not sufficient to
reliably reconstruct a complete prior model. The repository retained only the
latest JSON document, so geometry, topology, review state, validation state,
and calibrated measurements from an earlier revision could not be restored.
There were no API operations for undo, redo, or historical revision reads.

## Design

Each successful ordinary edit now:

1. Adds the current revision number to `undo_revision_stack`.
2. Clears `redo_revision_stack`, which makes a new post-undo edit an explicit
   history branch.
3. Produces the next monotonically increasing model revision as before.

Repositories archive the exact JSON snapshot for every saved revision. Undo
and redo load the revision at the top of the corresponding stack, restore its
complete model state, and save that state as a new revision. They do not move
the repository's current pointer backward and do not delete an abandoned
state. The current append-only `edit_history` is retained and receives a new
event containing:

- the restored snapshot revision;
- the revision that was left behind;
- the IDs of objects whose current state changed;
- actor and timestamp metadata.

An ordinary edit after undo clears the redo stack. Approved models reject both
ordinary edits and history restoration until an explicit transition back to
`in_review` occurs. Optimistic `expected_revision` checks apply to API history
commands exactly as they do to geometry and review commands.

The JSON repository uses content-preserving revision files under
`data/models/.revisions/<model-id>/<revision>.json` in production. Writes use
unique temporary paths followed by atomic replacement. Existing repositories
that predate revision archives create an archive for their current document on
the next successful save.

## API additions

- `GET /api/takeoff/models/{model_id}/revisions/{revision}` reads an immutable
  historical snapshot.
- `POST /api/takeoff/models/{model_id}/undo` restores the top undo snapshot.
- `POST /api/takeoff/models/{model_id}/redo` restores the top redo snapshot.

Undo and redo accept `expected_revision` and an optional `actor`. Empty stacks
return HTTP 422; stale clients and missing historical snapshots return HTTP
409. Revision path values must be positive.

## Files modified

- `apps/api/src/vision/domain/models.py`
- `apps/api/src/vision/domain/serialize.py`
- `apps/api/src/vision/domain/commands.py`
- `apps/api/src/vision/domain/repository.py`
- `apps/api/src/api/routes/takeoff_models.py`
- `tests/unit/vision/domain/test_domain_model.py`
- `tests/unit/vision/cv/test_api_route.py`

The editable model schema version is now `2.0.0-alpha.2`. Deserialization gives
older documents empty history stacks, so persisted alpha.1 models remain
readable.

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
  --output-dir evaluation_output\reviewed_changed_undo_redo
```

Results:

- Focused domain/API tests: 36 passed.
- Complete test suite: 223 passed, 2 dependency warnings.
- Compile check: passed.
- Whitespace/error check: `git diff --check` passed.
- JSON restart test: revisions 1-3 remained readable after repository
  re-instantiation; undo saved revision 4; no temporary files remained.
- API workflow test: stale undo was rejected; undo restored revision 2 as new
  revision 4; revision 3 remained readable; redo produced revision 5; a new
  edit after undo cleared redo and made redo return 422.
- Approval test: an approved model rejected undo; after explicit reopen, a
  subsequent edit could be undone while remaining `in_review`.

## Automatic first-pass regression result

This change does not alter extraction. The complete deterministic five-plan
benchmark exactly matches Cycle 3 and the previous approval baseline:

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
|---|---:|---:|---:|---:|---:|
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

Worst plan remains `degraded_dense` at 0.3531 macro IoU. Worst class/case
remains windows on `compact_thin` at 0.0000 IoU. The benchmark report is a
generated, intentionally untracked artifact at
`evaluation_output/reviewed_changed_undo_redo/report.json`.

## Assisted-workflow improvement

For any persisted correction represented by an edit command, one undo request
now restores the prior complete structured model and dependent quantities; one
redo request restores the abandoned result. No manual reconstruction or CV
rerun is required. The automated workflow test reaches the restored state in
one action and the redone state in one action. A real-user time-to-verified
measurement still requires the review interface and usability instrumentation.

## Remaining limitations

- The JSON repository is guarded by a process-local lock. Multiple API worker
  processes require a transactional database implementation.
- Snapshot storage is intentionally simple and currently stores a full JSON
  document per revision; compaction can be added after correctness is stable.
- Undo/redo is available for implemented commands only. Add/delete/split/merge
  commands for walls, openings, and rooms are still missing.
- Approval transitions are themselves recorded in history. Restoring the
  approval transition can freeze the restored state again, after which another
  explicit reopen is required.
- There is no review UI yet, so assisted completion time and click count have
  not been measured with a person.
- Native vector-PDF extraction, assemblies, waste factors, material costs, and
  dependency-aware cost recomputation remain future phases.

## Next highest-impact slice

Add constrained wall creation and deletion with graph-node reuse, dependency
warnings, validation recomputation, quantities derived from the resulting
model, and automatic undo/redo coverage. This turns the current endpoint-only
geometry editor into a workflow that can correct both missed and false wall
candidates--the two most consequential automatic wall failure modes.
