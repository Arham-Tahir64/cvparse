# FlowBuildr architecture assessment

Date: 2026-07-14

## Executive decision

FlowBuildr should be **partially reorganized around a persistent, geometry-first,
human-in-the-loop takeoff model**. The current CV implementation should remain
as a modular automatic candidate generator. Rewriting its working extraction
stages would discard useful behavior without solving the product's main gap:
there is no authoritative editable model, review workflow, dependency-aware
recalculation, or quantity/cost engine.

The migration should therefore introduce a domain layer between CV extraction
and presentation. CV output is imported as evidence-backed proposals. A user
reviews and edits a stable model. Rendering, validation, quantities, and costs
all consume that same model. Automatic reruns propose changes but never replace
confirmed manual work silently.

Current verification baseline:

- `190 passed` in 7.41 seconds with `pytest tests -q`.
- Five deterministic validation layouts cover thin/compact, asymmetric,
  dense/thick, skewed, and degraded input variants.
- The current worst automatic validation layout is `degraded_dense`, macro
  IoU 0.3531. The current worst class/case is `compact_thin` windows, IoU
  0.0000; `dense_large` doors are the next-lowest class/case at IoU 0.1047.
- The real ARCH-3 labeled plan scores wall 0.6831, door 0.2725, window 0.5789,
  room 0.8002, and macro IoU 0.5837.
- ARCH-5 is an unscored smoke plan because no matching ground truth exists. It
  produces 153 walls, 2 doors, 1 window, and 11 rooms; visual inspection shows
  low opening recall.

## 1. Existing architecture map

### Runtime flow

```text
Upload or CLI path
    -> MIME/suffix handling
    -> PDF page rasterization or raster decode
    -> preprocessing and deskew
    -> structural ROI selection
    -> first line/OCR proposal pass
    -> line classification
    -> drafting removal
    -> second line/classification pass
    -> wall extraction
    -> junction snapping
    -> preliminary room extraction
    -> door detection and wall splitting
    -> final room extraction
    -> window detection
    -> semantic raster masks
    -> room OCR labeling
    -> CVTakeoffResult
    -> JSON / annotation adapter / SVG / annotated PDF
```

The orchestrator is `vision.cv.pipeline.run_pipeline_state`. Stages mutate a
shared `PipelineState`. `PipelineState.to_takeoff_result()` copies detected
entities into an ephemeral `CVTakeoffResult`.

### Input loading

- `preprocessing.load_image` accepts PDF, PNG, JPEG, TIFF, and BMP bytes.
- PDFs are opened with PyMuPDF and immediately rasterized at `working_dpi`
  (200 DPI by default).
- Multi-page selection is explicit or uses low-resolution pixel variance.
- Raster inputs are decoded by OpenCV.
- Deskew uses the minimum-area rectangle of all foreground pixels and only
  applies rotations within five degrees.
- There is no perspective correction, drawing-region confirmation, or
  persistent input calibration record.

### Vector preservation

Useful vector PDF content is currently discarded during ingestion. Lines,
curves, text objects, layers, repeated blocks, stroke widths, and PDF object
identity are not extracted. The original PDF is reused only as the visual
background for the annotated-PDF exporter. It is not a geometry source.

### Automatic extraction

- Walls: LSD strokes are classified, drafting pixels are removed, parallel
  faces are paired into centerlines, thickness is sampled, and junctions are
  snapped. A later semantic stage rebuilds wall regions from wall geometry,
  room support, measurement vetoes, topology recovery, and optional exterior
  rectangle reconstruction.
- Doors: Hough circle proposals identify swing arcs. Candidates are checked
  against wall continuation, opening support, leaf ink, hinge position, and
  preliminary room topology. Accepted doors split host walls.
- Windows: local repeated-frame lines and face gaps are checked against a host
  wall, exterior room context, shell thickness, door conflicts, and duplicate
  overlap.
- Rooms: preliminary and final passes derive free-space components and room
  polygons. Openings are temporarily bridged as barriers. OCR assigns labels.
- Semantic masks: wall, door, window, and room ownership is rasterized after
  object detection for evaluation and PDF export.

### Scale handling

- Working DPI is normalized for PDF rendering but a drawing scale is never
  detected, requested, stored, or confirmed.
- `Wall.length_ft` exists but is never calculated.
- Room area remains `area_px`; opening widths and wall thicknesses remain
  pixels.
- One adaptive door threshold uses a wall-thickness distribution, and several
  later heuristics use ratios, but most thresholds are fixed pixels centralized
  in `PipelineConfig`.
- Important fixed assumptions include morphology kernels, line lengths and
  gaps, wall thickness range, snapping radii, door radius range, window length
  range, room barrier closing, minimum room area, OCR distance, and semantic
  plan margins.

### Representation and export

- Value objects: `Point`, `LineSegment`, and `TextElement`.
- Entities: `Wall`, `Junction`, `Gap`, `Door`, `Window`, and `Room`.
- IDs are deterministic counters only within one run. They are not stable
  across reprocessing or persistence.
- Wall provenance and some confidence fields exist. Door confidence exists.
  Window/room structural confidence and review provenance do not.
- Doors and windows reference walls, but openings are duplicated across
  `Gap`, `Door`, and `Window` rather than represented by one authoritative
  opening entity.
- Rooms lack boundary-wall, opening, adjacency, perimeter, and calibrated-area
  relationships.
- `serialize.py` emits schema 1.0.0. `annotation_adapter.py` adds hard-coded
  `pending` states but has no update path.
- The API exposes only `POST /api/cv/takeoff`; results are not persisted.
- PDF rendering consumes both `CVTakeoffResult` and separate semantic masks
  from `PipelineState`. Consequently visualization already has two sources of
  truth rather than being derived solely from the structured result.

### Editing and takeoff propagation

There are no edit endpoints, commands, persistence repositories, UI assets,
undo/redo, audit events, locks, precedence rules, or dependency graph. There is
also no material quantity or cost calculation implementation. The endpoint is
named `takeoff`, but it currently returns detection geometry only.

## 2. Main architectural weaknesses

1. **Raster-only PDF ingestion.** Native vector evidence is lost before CV.
2. **No calibrated coordinate system.** Pixel geometry cannot support reliable
   real-world quantities.
3. **Ephemeral state.** Objects exist only during one request and IDs are not
   stable across runs.
4. **No authoritative opening model.** Gaps, doors, and windows can diverge.
5. **Incomplete topology.** Junctions exist, but walls do not store stable start
   and end nodes, connected walls, or opening relationships.
6. **Incomplete room model.** Rooms lack shared boundaries, adjacency,
   perimeter, calibrated area, and dependency links.
7. **No review semantics.** `pending` is an adapter constant, not domain state.
8. **No uncertainty/issue model.** A confidence number is not converted into an
   actionable, impact-ranked review queue.
9. **No edit commands or invariants.** Nothing constrains an opening to a host
   wall or preserves shared room geometry during edits.
10. **No recomputation graph.** An edit cannot invalidate and selectively
    refresh derived geometry or quantities.
11. **No takeoff engine.** Length, area, count, waste, material, and cost
    calculations are absent.
12. **Multiple sources of truth.** Renderers may use object lists and transient
    masks that are not serializable as one corrected model.
13. **Fixed-pixel sensitivity.** Centralization is good, but thresholds remain
    tied to a nominal 200-DPI, quarter-scale drawing.
14. **Manhattan-first assumptions.** Rectilinear snapping and exterior rectangle
    reconstruction can be useful but need explicit plan capabilities and review
    flags for non-Manhattan geometry.
15. **Tolerant CLI failure mode.** The CLI can export partial results after a
    stage error without encoding model completeness or approval readiness.

## 3. Reusable components

Retain and adapt:

- MIME validation, raster decoding, PDF rendering fallback, deskew, and debug
  artifact production.
- The staged `PipelineState` orchestrator as an automatic candidate pipeline.
- Line detection, OCR abstraction, drafting removal, paired-wall extraction,
  junction snapping, room extraction, and existing opening evidence checks.
- Geometry helpers and immutable point/segment value types.
- Wall source provenance, merge confidence, and detector debug metrics.
- Deterministic multi-layout validation and per-class reporting.
- The output adapter boundary, which is the correct seam for introducing a new
  domain model without breaking schema 1.0.0 immediately.
- PDF/SVG rendering mechanics, after they are changed to consume only the
  authoritative model.
- Existing tests as behavioral characterization before replacement.

## 4. Components requiring refactoring or addition

### Refactor

- Split input ingestion into `input inspection`, `vector extraction`, and
  `raster fallback/normalization`.
- Convert CV entities into proposal/evidence objects rather than final takeoff
  objects.
- Replace run-local IDs at the domain boundary with persistent UUID-like IDs
  and source fingerprints for rerun matching.
- Introduce one `Opening` entity; door/window semantics attach to it.
- Promote junctions to stable wall-graph nodes and store wall connectivity.
- Derive room boundaries and adjacency from the corrected wall/opening graph.
- Make renderers derive masks from the authoritative domain model.
- Version the API contract while preserving schema 1.0.0 through a legacy
  adapter.

### Add

- Plan/session persistence and repositories.
- Scale calibration with source, confidence, and confirmation status.
- Review status, source, confidence breakdown, validation issues, and locks.
- Edit commands, constraint validation, undo/redo, and audit events.
- Dependency-aware recomputation.
- Quantity/cost calculation and material assemblies.
- Review and edit API endpoints, then a constrained review UI.
- Vector-PDF inspection and evidence extraction.
- Assisted-completion benchmarks and ground-truth takeoff fixtures.

## 5. Proposed editable data model

All persisted entities carry `id`, `revision`, `source`, `review_status`,
`locked`, `created_at`, and `updated_at`. Coordinates are stored in a plan-local
coordinate system; calibrated values are derived through `ScaleCalibration`.

```text
TakeoffModel
  id, schema_version, revision, source_document, active_page
  coordinate_system, scale_calibration, drawing_region
  nodes, walls, openings, doors, windows, rooms
  validation_issues, review_queue, quantities, costs, edit_history
  approval_status

ScaleCalibration
  pixels_per_unit, unit, method, evidence, confidence, review_status

Node
  id, point, connected_wall_ids, kind, confidence, source, review_status

Wall
  id, start_node_id, end_node_id, centerline, polygon, thickness
  wall_type, connected_wall_ids, opening_ids
  length_px, length, confidence, source, review_status

Opening
  id, wall_id, start_offset, end_offset, center, width
  orientation, classification (door/window/archway/unknown)
  confidence, source, review_status

Door
  id, opening_id, subtype, swing_direction, hinge_side
  confidence, source, review_status

Window
  id, opening_id, subtype, sill_height
  confidence, source, review_status

Room
  id, polygon, label, boundary_wall_ids, opening_ids
  neighboring_room_ids, area, perimeter
  confidence, source, review_status

ValidationIssue
  id, code, severity, affected_object_ids, evidence
  uncertainty, structural_impact, cost_impact, priority, status

EditEvent
  id, model_revision_before, model_revision_after, command
  inverse_command, affected_object_ids, actor, timestamp
```

Opening positions should be stored as offsets along the host wall. This keeps
them attached when the wall moves and makes deduplication and width validation
natural. Door swing geometry is secondary presentation/classification data;
the opening center and width are authoritative.

## 6. Proposed confidence and review system

Use a confidence breakdown, not only a scalar:

```text
geometry_quality
wall_or_room_association
scale_plausibility
symbol_evidence
topology_consistency
vector_raster_agreement
duplicate_conflict
```

The combined confidence remains visible but every low score can explain itself.
Domain review states are:

```text
confirmed
likely_correct
needs_review
conflicting
unknown
rejected
```

Automatic candidates enter `likely_correct`, `needs_review`, `conflicting`, or
`unknown`; never `confirmed`. Structural validators generate explicit issues
for unclosed rooms, floating openings, duplicate openings, disconnected walls,
invalid polygons, implausible scale, and inconsistent quantities.

Queue priority is computed from normalized components:

```text
priority = uncertainty * structural_impact * max(cost_impact, impact_floor)
```

Scale uncertainty, exterior-wall gaps, invalid room topology, and large
quantity changes rank ahead of minor label or visualization issues. Queue items
must include the affected crop/objects, reason, suggested action, and downstream
quantity impact.

## 7. Proposed edit-propagation strategy

Edits are commands applied transactionally to a model revision. Each command:

1. validates preconditions and geometric constraints;
2. writes the smallest authoritative geometry change;
3. records an inverse command and audit event;
4. marks dependent artifacts dirty;
5. recomputes only affected topology, rooms, quantities, costs, and issues;
6. returns the changed objects and new revision.

Dependency examples:

```text
scale -> every calibrated measurement -> every quantity -> every cost
wall nodes/thickness -> wall polygon/length -> adjacent rooms/openings
opening -> host wall net area -> door/window counts -> materials/costs
room boundary -> area/perimeter -> flooring/ceiling/trim -> costs
room merge/split -> adjacency/shared boundaries -> room assemblies
```

Wall endpoint edits invalidate the wall, incident nodes/walls, hosted openings,
adjacent rooms, and their quantities. Opening edits invalidate only the host
wall's net areas, adjacent-room connectivity, relevant counts, and costs. A
full CV rerun is not an edit-propagation mechanism.

## 8. Automatic-versus-manual precedence

1. Confirmed manual geometry and classifications are authoritative and locked.
2. Manual unconfirmed edits outrank automatic proposals but remain reviewable.
3. Automatic reruns operate only on unlocked objects/regions.
4. Rerun results are matched by source fingerprint and geometric similarity;
   stable domain IDs are preserved when matched.
5. Unmatched automatic proposals are added as candidates, not confirmed data.
6. Conflicts with locked/manual objects create `ValidationIssue` records and
   side-by-side proposals; they never overwrite silently.
7. Rejected proposals remain auditable and are not recreated unchanged on each
   rerun.
8. A user may explicitly unlock or reset an object to automatic mode.
9. Approval freezes the exact model revision used for quantities and export.

## 9. Phased migration plan

### Phase 1: structured source of truth

- Add the domain model, review/source metadata, stable IDs, validation issues,
  revisions, and scale record.
- Add a compatibility importer from `CVTakeoffResult`.
- Preserve schema 1.0.0 through an adapter while adding a versioned model
  serializer.
- Move annotation generation to the domain model.
- Add persistence behind a repository interface.

Exit criterion: a generated plan can be saved, loaded without loss, edited as
domain objects, and rendered from the reloaded model.

### Phase 2: validation and confidence

- Add wall graph, opening, room, scale, duplicate, and quantity validators.
- Produce explainable confidence breakdowns and an impact-ranked review queue.

Exit criterion: uncertain or inconsistent CV results are explicit and cannot
be approved without resolution or intentional acknowledgement.

### Phase 3: human correction tools

- Add command APIs for wall, opening, door/window, room, label, and scale edits.
- Implement constraints, snapping, confirmation, rejection, and issue handling.
- Add a review UI ordered by scale, walls/topology, rooms, openings, labels,
  then quantity impact.

Exit criterion: the representative plans can be corrected without direct JSON
or code edits, with actions and time measured.

### Phase 4: dependency-aware recalculation

- Add dirty dependency tracking, targeted room/topology recomputation, takeoff
  assemblies, quantities, waste, and costs.
- Add undo/redo and audit-history persistence.

Exit criterion: every supported edit updates exactly the dependent quantities,
and undo/redo reproduces prior model and takeoff revisions.

### Phase 5: generalization improvements

- Add vector-PDF evidence and hybrid input handling.
- Replace fixed pixels with plan-scale/line-thickness-derived values.
- Continue cross-layout automatic extraction improvements using the existing
  validation corpus plus additional real labeled plans.

Exit criterion: first-pass metrics improve across all representative plans
without increasing assisted completion effort or regressions.

### Phase 6: review efficiency

- Measure and optimize time, clicks, accepted-auto percentage, common edits,
  final takeoff error, and highest-cost-impact errors.

Exit criterion: `time_to_verified_takeoff` and final quantity error meet agreed
product targets on the representative corpus.

## 10. Smallest high-impact first implementation step

Introduce a **versioned editable domain model plus a lossless CV-result
importer**, without changing the detectors yet.

This first slice should include:

- `TakeoffModel`, `PlanSource`, `ScaleCalibration`, `ReviewStatus`,
  `ObjectSource`, `ConfidenceBreakdown`, `ValidationIssue`, `Node`, `Wall`,
  `Opening`, `Door`, `Window`, and `Room` dataclasses;
- stable domain IDs distinct from run-local detector IDs;
- source detector IDs/provenance retained on imported objects;
- one authoritative opening per imported gap, with door/window records linked
  to it and host walls;
- a deterministic `CVTakeoffResult -> TakeoffModel` importer;
- a versioned JSON round trip;
- structural validation for dangling references and duplicate opening ranges;
- compatibility tests proving the existing schema and API still work.

Why this first: every later requirement—manual edits, review states, confidence,
undo/redo, selective recalculation, quantities, costs, and authoritative
rendering—depends on a stable model. It is independently testable, does not
weaken current CV behavior, and creates a safe adapter seam for incremental
migration.

## Validation strategy during migration

Evaluate two modes separately:

- **Automatic first pass:** existing per-class precision, recall, F1, IoU,
  counts, position/orientation error, wall-length error, room-area error, scale
  error, and worst plan/class.
- **Assisted completion:** review flags, accepted automatic objects, edit action
  count, clicks, elapsed correction time, final quantity error, and
  `time_to_verified_takeoff`.

Until more real labels exist, synthetic fixtures can prove invariants and edit
propagation but cannot establish product accuracy. ARCH-3 remains the only real
scored plan; ARCH-5 remains a qualitative smoke case. Additional labeled vector,
hybrid, scanned, skewed, low-resolution, and complex plans are required before
claiming consistent assisted-completion performance.
