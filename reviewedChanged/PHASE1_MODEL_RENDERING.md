# Phase 1 authoritative model-rendering log

Date: 2026-07-14

## Goal

Remove the source-of-truth split for reviewed annotation overlays. A persisted
model that has been edited and reloaded must export current wall, opening,
door, window, and room geometry without consulting legacy CV results or masks.

## Implemented

- Added a domain annotation adapter that emits:
  - model/schema/revision and source-image metadata;
  - authoritative wall centerlines and polygons;
  - calibrated and pixel measurements;
  - opening-backed door and window geometry;
  - room polygons, labels, measurements, and relationships;
  - actual confidence, review state, lock state, source kind, and object
    revision;
  - stable domain IDs and graph/opening relationships.
- Rejected walls and their dependent openings/symbols are omitted from current
  output. Other uncertain states stay visible and explainable.
- Unclassified openings remain visible as opening elements instead of being
  silently discarded.
- Added a deterministic, transparent SVG renderer driven only by
  `TakeoffModel`:
  - rooms render beneath structure;
  - wall footprints use authoritative polygons;
  - openings use their host-wall offsets;
  - door leaves/arcs and windows render from current opening relationships;
  - model ID, model revision, stable element IDs, source kind, review state,
    and object revision are embedded as data attributes;
  - room labels are XML escaped;
  - review/conflict styling remains separate from class colors.
- Added persisted-model endpoints:
  - `GET /api/takeoff/models/{model_id}/annotations`
  - `GET /api/takeoff/models/{model_id}/overlay.svg`
- SVG responses include an ETag containing model ID and revision.

The existing CV annotation document and PDF renderer remain unchanged for
backward compatibility.

## Files added or changed

- `apps/api/src/api/routes/takeoff_models.py`
- `apps/api/src/vision/adapters/domain_annotation_adapter.py`
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
  --output-dir evaluation_output\reviewed_changed_model_render
```

Results:

- Focused domain/API suite: 24 passed.
- Complete suite: 211 passed.
- SVG output parses as XML and is byte-identical before and after a lossless
  JSON model reload.
- The API integration test persists a model, moves a shared endpoint, fetches
  both new exports, and proves that the edited coordinates, model revision,
  stable wall ID, review state, and SVG ETag all come from revision 2.
- A rejected wall is absent from the annotation document rather than rendered
  from stale detector state.
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

- The transparent overlay is authoritative, but the uploaded source PDF/image
  is not yet persisted. A combined reviewed PDF therefore cannot be regenerated
  after a service restart without the caller supplying the source again.
- Legacy `POST /api/cv/takeoff` annotations still describe the immediate CV
  result because that request may be non-persistent. Persisted review workflows
  should use the new model endpoints.
- Pixel masks are intentionally not reused: they cannot represent subsequent
  vector edits. The model renderer currently uses wall polygons and opening
  geometry, so it does not reproduce every raster contour artifact.
- Material quantities and costs are not yet implemented, so rendering and
  geometry now share a source but takeoff calculations are still incomplete.

The next slice should add source-asset persistence keyed by the existing upload
fingerprint and a combined PDF/image export that overlays this same SVG/domain
geometry. After that, implement the first model-native quantity summary so the
rendered and calculated outputs demonstrably share one revision.
