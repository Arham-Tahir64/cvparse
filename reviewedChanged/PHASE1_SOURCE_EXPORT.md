# Phase 1 source persistence and reviewed export log

Date: 2026-07-15

## Goal

Persist the original plan beside the editable model and generate a combined
reviewed PDF from the original source plus exactly the current model revision.
This closes the remaining annotation source split after a service restart.

## Implemented

- Added a content-addressed source-asset repository with in-memory and file
  implementations.
- Source assets are keyed by the upload SHA-256 already stored in `PlanSource`.
- Every save and load verifies that the bytes match the fingerprint. Empty,
  malformed, substituted, or tampered content fails explicitly.
- File writes use unique temporary names, atomic replacement, and cleanup.
  Saving the same upload is idempotent; different bytes cannot replace a known
  fingerprint.
- `POST /api/cv/takeoff` with `persist_model=true` now stores the original bytes
  before saving the model. A duplicate model import remains a revision conflict,
  while its identical source save remains safe.
- Runtime storage defaults to `data/sources` and can be changed with
  `FLOWBUILDR_SOURCE_DIR`. Both source and model runtime directories are now
  ignored by Git.
- Added a model-native combined PDF renderer:
  - native PDF input preserves the selected original page via `insert_pdf`;
  - raster input preserves the source pixels on a correctly scaled PDF page;
  - rooms, wall polygons, hosted openings, and door geometry come only from the
    current `TakeoffModel`;
  - rejected objects are omitted;
  - the output metadata records model ID and revision.
- Added:

```text
GET /api/takeoff/models/{model_id}/reviewed.pdf
```

The response ETag and filename include the current model revision. Missing
source assets return 404, invalid source content returns 422, and persisted
integrity failures return 500 rather than serving untrusted bytes.

## Files added or changed

- `.gitignore`
- `apps/api/src/api/model_store.py`
- `apps/api/src/api/routes/cv_takeoff.py`
- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/adapters/domain_pdf.py`
- `apps/api/src/vision/domain/source_assets.py`
- `tests/unit/vision/cv/test_api_route.py`
- `tests/unit/vision/domain/test_domain_model.py`

## Verification

Commands:

```powershell
$env:PYTHONPATH = "apps/api/src"
.\.venv\Scripts\python.exe -m compileall -q apps\api\src
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools\validate_generalization.py `
  --output-dir evaluation_output\reviewed_changed_source_export
```

Results:

- Focused source/render/API suite: 29 passed.
- Complete suite: 216 passed.
- The API test performs upload persistence, shared endpoint correction,
  revision-2 quantity calculation, and revision-2 combined PDF export in one
  workflow.
- File-repository tests prove idempotent save, one stored binary, no remaining
  temp files, exact reload, and tamper rejection.
- Native-PDF rendering preserves original vector text, one source page, current
  revision metadata, and authoritative model drawings.
- Raster API rendering returns a readable one-page PDF with revision ETag and
  vector overlays.
- The five-plan validation summary and every per-class metric are exactly equal
  to `evaluation_output/generalization_cycle3/report.json`.

| Plan | Wall IoU | Door IoU | Window IoU | Room IoU | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact_thin | 0.6086 | 0.3774 | 0.0000 | 0.9564 | 0.4856 |
| asymmetric_medium | 0.6815 | 0.3177 | 0.3758 | 0.9661 | 0.5853 |
| dense_large | 0.4398 | 0.1047 | 0.5345 | 0.7030 | 0.4455 |
| skewed_medium | 0.4669 | 0.3683 | 0.1150 | 0.9309 | 0.4703 |
| degraded_dense | 0.2786 | 0.1533 | 0.2783 | 0.7022 | 0.3531 |

## Visual PDF QA

The PDF workflow was used to generate and rasterize a temporary combined
native-PDF fixture at 200 DPI. Visual inspection confirmed:

- original source text remained readable and separate from the overlay;
- the manually moved shared node produced the expected slanted connected walls;
- room fill remained below structural geometry;
- wall, door, and window colors remained distinct;
- no clipping, black fill artifacts, broken paths, or unreadable glyphs.

The bundled Poppler wrapper was present but could not launch because its
external-execution approval timed out. The same final artifact was rendered
with PyMuPDF and inspected at original resolution. All temporary QA artifacts
were removed afterward.

## Remaining limitations and next slice

- Source/model creation is ordered and safe but not one database transaction;
  a failed model save can leave an unreferenced content-addressed source asset.
  This is harmless but needs garbage collection or a transactional database at
  multi-user scale.
- Only the selected page is exported. Multi-page reviewed packages and source
  download authorization are not implemented.
- The combined PDF uses authoritative vector domain geometry, not stale raster
  masks. It therefore does not reproduce pixel-level contour artifacts.
- Vector PDF content is preserved for export, but native vector extraction for
  automatic candidate generation is still not implemented.
- Approval, undo/redo, and most constrained geometry edit commands remain.

The next slice should implement model approval/reopen plus undo/redo so a
reviewed revision can be frozen, traced, reversed, and reapproved without
rerunning CV or losing the persisted source.
