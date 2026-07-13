"""PipelineConfig: every tunable parameter in the CV pipeline.

No bare numeric literals appear in module code; everything routes through here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

DEFAULT_ROOM_LABEL_VOCAB: tuple[str, ...] = (
    "KITCHEN", "BEDROOM", "BED", "BATH", "BATHROOM", "ENSUITE", "LIVING",
    "DINING", "FAMILY", "OFFICE", "DEN", "STUDY", "LAUNDRY", "MUDROOM",
    "FOYER", "ENTRY", "HALL", "HALLWAY", "CLOSET", "WIC", "PANTRY",
    "GARAGE", "STAIRS", "STAIR", "POWDER", "STORAGE", "MECHANICAL",
    "MECH", "PRIMARY", "MASTER BEDROOM", "MASTER", "GUEST SUITE",
    "GUEST", "GYM/YOGA", "GYM", "YOGA", "REC ROOM AREA", "REC ROOM",
    "RECREATION", "LINEN", "LNDRY",
)


@dataclass(slots=True)
class PipelineConfig:
    # --- Global ---
    working_dpi: int = 200                     # working resolution for all modules
    manhattan: bool = True                     # snap near-axis walls to exact H/V
    generate_preview: bool = False             # include base64 preview in result
    debug_visualize: bool = False              # write per-stage debug images
    debug_output_dir: Optional[str] = None     # where debug images go

    # --- Module 01: Raster Preprocessing ---
    binarization_method: Literal["otsu", "adaptive"] = "adaptive"
    adaptive_block_size: int = 25              # must be odd; corrected if even
    adaptive_c: int = 12
    deskew_max_angle_deg: float = 5.0          # skip rotation beyond this angle
    denoise_h: float = 3.0                     # fastNlMeansDenoising strength
    min_component_area_px: int = 4             # remove smaller connected components

    # --- Module 02: Structural ROI ---
    roi_h_kernel_len: int = 24                 # horizontal kernel length for wall seeds
    roi_v_kernel_len: int = 24
    roi_close_kernel_px: int = 21              # closes the seed to fill gaps
    roi_dilate_kernel_px: int = 61             # expands ROI to include nearby openings
    roi_min_component_area_frac: float = 0.004 # minimum component size to treat as plan
    roi_fallback_full_image: bool = True       # use full image if no structural seed found

    # --- Module 03: Line Detection ---
    lsd_scale: float = 0.8
    lsd_sigma_scale: float = 0.6
    min_line_length_px: float = 15.0
    line_merge_angle_tol_deg: float = 2.0
    line_merge_perpendicular_tol_px: float = 2.0
    line_merge_gap_tol_px: float = 5.0
    line_merge_bridge_fill_min: float = 0.55   # min wall-pixel fill in a gap before bridging

    # --- Module 04: Line Filtering ---
    # LSD reports 1-4 px widths for both hairlines and wall-face strokes; this
    # threshold only needs to exclude genuinely thick strokes
    dimension_line_max_thickness_px: float = 2.5
    text_proximity_px: float = 12.0
    text_bbox_pad_px: float = 4.0              # padding around text bboxes for rule 1
    leader_max_length_px: float = 60.0         # rule 2 length ceiling
    dimension_min_length_px: float = 30.0      # rule 3 length floor
    dimension_tick_search_px: float = 8.0      # tick-mark search radius at endpoints
    dimension_text_dist_px: float = 20.0       # max dist from midpoint to dimension text
    hatch_cluster_density_threshold: int = 8
    hatch_short_line_max_px: float = 30.0
    hatch_neighbor_radius_px: float = 40.0     # midpoint radius for hatch clustering
    hatch_angle_tol_deg: float = 5.0           # parallelism tolerance for hatch neighbors
    hatch_diagonal_cluster: bool = True        # rule 4b: dense parallel diagonals = hatch
    grid_dash_min_gap_px: float = 4.0
    grid_min_length_px: float = 200.0          # rule 5 length floor
    grid_min_gap_runs: int = 3                 # rule 5 minimum background runs

    # --- Module 05: Wall Extraction ---
    wall_thickness_min_px: float = 5.0
    wall_thickness_max_px: float = 80.0        # also max face separation for pairing
    wall_min_length_px: float = 45.0
    parallel_angle_tolerance_deg: float = 3.0
    parallel_overlap_min_ratio: float = 0.45
    parallel_overlap_min_px: float = 36.0
    face_support_sample_step_px: int = 10
    face_support_window_pad_px: int = 4
    face_support_min_run_px: float = 30.0
    single_face_min_thickness_px: float = 5.0  # math.inf disables single-face walls
    thin_branch_min_thickness_px: float = 3.0
    thin_branch_min_length_px: float = 30.0
    thin_branch_max_length_px: float = 120.0
    thin_branch_orthogonal_support_dist_px: float = 40.0
    # candidates shorter than this skip the orthogonal-support check; 0 disables
    # the bypass (arc chords and other debris otherwise slip through as walls)
    thin_branch_stub_bypass_length_px: float = 0.0
    thin_branch_max_overlap_ratio: float = 0.30      # max overlap with an accepted wall
    visual_thickness_search_px: int = 30
    visual_thickness_max_px: float = 45.0
    visual_thickness_endpoint_margin_px: float = 40.0
    manhattan_snap_angle_deg: float = 5.0

    # --- Module 06: Junction Snapping ---
    junction_snap_radius_px: float = 12.0
    gap_closure_max_px: float = 8.0            # must be < junction_snap_radius_px
    gap_closure_max_iterations: int = 10
    junction_coincidence_tol_px: float = 0.5   # invariant check tolerance
    zero_length_wall_px: float = 2.0           # walls shorter than this are removed

    # --- Module 07: Doors ---
    door_arc_min_radius_px: float = 20.0
    # At 1/4" = 1'-0" and 200 DPI a common 2'-6" leaf is ~125 px.
    door_arc_max_radius_px: float = 160.0
    door_wall_snap_px: float = 15.0
    hough_circles_dp: float = 1.0
    hough_circles_min_dist: float = 30.0
    hough_circles_param1: float = 100.0        # Canny upper threshold
    hough_circles_param2: float = 25.0         # accumulator threshold; tune 20-50
    arc_coverage_min: float = 0.15
    arc_coverage_max: float = 0.40
    door_dedup_dist_px: float = 10.0           # hinge distance for deduplication
    door_split_t_min: float = 0.1              # split only if hinge parameter in (min, max)
    door_split_t_max: float = 0.9
    wall_erase_extra_px: int = 4               # extra thickness when erasing walls

    # --- Module 08: Windows ---
    window_gap_min_px: float = 25.0
    window_gap_max_px: float = 220.0
    window_gap_scan_half_band_px: int = 5
    window_gap_side_sample_px: int = 12
    window_gap_min_side_fill: float = 0.18
    window_inner_line_perp_frac: float = 0.6   # max perp dist from centerline / wall thickness
    window_inner_line_angle_tol_deg: float = 5.0
    window_merge_overlap_ratio: float = 0.5    # candidates overlapping more than this merge
    window_scan_min_samples: int = 20

    # --- Module 09: Rooms ---
    min_room_area_px: float = 1000.0
    max_room_area_frac: float = 0.85           # faces above this fraction = outer face
    enable_floodfill_fallback: bool = True
    floodfill_wall_dilation_px: int = 4
    planarity_repair_max_iterations: int = 20
    room_poly_epsilon_frac: float = 0.01       # approxPolyDP epsilon / contour perimeter
    semantic_room_min_seeds: int = 2
    room_barrier_min_line_px: int = 30
    room_barrier_gap_close_px: int = 170
    room_barrier_thickness_px: int = 7
    semantic_room_poly_epsilon_frac: float = 0.002
    semantic_room_seed_confidence: float = 0.6
    semantic_plan_margin_px: int = 90

    # --- Module 10: OCR ---
    ocr_engine: Literal["paddle", "tesseract"] = "paddle"
    ocr_first_pass_confidence: float = 0.3
    ocr_second_pass_confidence: float = 0.6
    room_label_vocab: tuple[str, ...] = field(default=DEFAULT_ROOM_LABEL_VOCAB)
    label_room_max_distance_px: float = 80.0
