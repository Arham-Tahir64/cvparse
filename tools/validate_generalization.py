"""Run the CV pipeline on deterministic, exactly labelled floor-plan probes.

The repository has one real ground-truth overlay. These generated cases add
independent regression coverage for layouts, scale, wall thickness, opening
placement, skew, and raster degradation without treating pipeline predictions
as labels. They are deliberately generic architectural primitives; no fixture
coordinates are used by production code.

Usage:
    PYTHONPATH=apps/api/src python tools/validate_generalization.py
    PYTHONPATH=apps/api/src python tools/validate_generalization.py \
        --cases compact_thin,asymmetric_medium --output-dir evaluation_output/generalization
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

from vision.cv.config import PipelineConfig  # noqa: E402
from vision.cv.pipeline import run_pipeline_state  # noqa: E402


CLASSES = ("wall", "door", "window", "room")
CLASS_COLOURS = {
    "wall": (41, 38, 214),
    "door": (43, 161, 43),
    "window": (180, 119, 31),
    "room": (232, 199, 173),
}


@dataclass(frozen=True)
class SyntheticPlan:
    name: str
    image: np.ndarray
    truth: dict[str, np.ndarray]
    truth_objects: dict[str, int]
    variation: tuple[str, ...]


WallSpec = tuple[int, int, int, int]
DoorSpec = tuple[tuple[int, int], int, int, tuple[int, int], tuple[int, int], int]
WindowSpec = tuple[tuple[int, int], int, str]
RoomSpec = tuple[int, int, int, int, str]


def _render_plan(
    name: str,
    size: tuple[int, int],
    wall_thickness: int,
    walls: list[WallSpec],
    doors: list[DoorSpec],
    windows: list[WindowSpec],
    rooms: list[RoomSpec],
    variation: tuple[str, ...],
    clutter: bool = True,
) -> SyntheticPlan:
    width, height = size
    image = np.full((height, width), 255, np.uint8)
    truth = {key: np.zeros((height, width), np.uint8) for key in CLASSES}

    for x1, y1, x2, y2 in walls:
        cv2.line(image, (x1, y1), (x2, y2), 0, wall_thickness)
        cv2.line(truth["wall"], (x1, y1), (x2, y2), 255, wall_thickness)

    def erase_opening(start: tuple[int, int], end: tuple[int, int]) -> None:
        cv2.line(image, start, end, 255, wall_thickness + 12)
        cv2.line(truth["wall"], start, end, 0, wall_thickness + 12)

    for hinge, closed_deg, leaf_deg, opening_start, opening_end, radius in doors:
        erase_opening(opening_start, opening_end)
        arc_start, arc_end = sorted((closed_deg, leaf_deg))
        cv2.ellipse(
            image, hinge, (radius, radius), 0, arc_start, arc_end, 0,
            max(2, round(wall_thickness * 0.14)),
        )
        leaf_angle = math.radians(leaf_deg)
        leaf = (
            round(hinge[0] + radius * math.cos(leaf_angle)),
            round(hinge[1] + radius * math.sin(leaf_angle)),
        )
        cv2.line(image, hinge, leaf, 0, max(2, round(wall_thickness * 0.14)))
        delta = (leaf_deg - closed_deg + 180) % 360 - 180
        polygon = [hinge]
        for degree in np.linspace(closed_deg, closed_deg + delta, 32):
            angle = math.radians(float(degree))
            polygon.append((
                round(hinge[0] + radius * math.cos(angle)),
                round(hinge[1] + radius * math.sin(angle)),
            ))
        cv2.fillPoly(truth["door"], [np.asarray(polygon, np.int32)], 255)

    for (cx, cy), span, orientation in windows:
        half = span // 2
        frame_offset = max(3, round(wall_thickness * 0.32))
        frame_thickness = max(1, round(wall_thickness * 0.09))
        if orientation == "H":
            erase_opening((cx - half, cy), (cx + half, cy))
            for offset in (-frame_offset, 0, frame_offset):
                cv2.line(
                    image, (cx - half, cy + offset), (cx + half, cy + offset),
                    0, frame_thickness,
                )
            cv2.rectangle(
                truth["window"],
                (cx - half, cy - wall_thickness // 2),
                (cx + half, cy + wall_thickness // 2), 255, -1,
            )
        else:
            erase_opening((cx, cy - half), (cx, cy + half))
            for offset in (-frame_offset, 0, frame_offset):
                cv2.line(
                    image, (cx + offset, cy - half), (cx + offset, cy + half),
                    0, frame_thickness,
                )
            cv2.rectangle(
                truth["window"],
                (cx - wall_thickness // 2, cy - half),
                (cx + wall_thickness // 2, cy + half), 255, -1,
            )

    label_scale = max(0.55, min(1.05, wall_thickness / 22.0 * 0.8))
    label_thickness = max(1, round(wall_thickness * 0.09))
    for x1, y1, x2, y2, label in rooms:
        cv2.rectangle(truth["room"], (x1, y1), (x2, y2), 255, -1)
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness,
        )[0]
        tx = round((x1 + x2 - text_size[0]) / 2)
        ty = round((y1 + y2 + text_size[1]) / 2)
        cv2.putText(
            image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
            label_scale, 0, label_thickness, cv2.LINE_AA,
        )

    if clutter:
        # Dimension chain outside the plan plus a leader through open space.
        margin_y = max(20, min(y for _, y, _, _ in walls) - 55)
        xs = sorted({x for x1, _, x2, _ in walls for x in (x1, x2)})
        if len(xs) >= 2:
            cv2.line(image, (xs[0], margin_y), (xs[-1], margin_y), 0, 1)
            for x in xs:
                cv2.line(image, (x, margin_y - 12), (x, margin_y + 12), 0, 1)
        y_mid = height // 2
        cv2.arrowedLine(
            image, (width // 5, y_mid), (2 * width // 5, y_mid),
            0, 1, tipLength=0.05,
        )
        cv2.putText(
            image, "12'-6\"", (width // 4, y_mid - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA,
        )

    return SyntheticPlan(
        name=name, image=image, truth=truth,
        truth_objects={
            "wall": len(walls), "door": len(doors),
            "window": len(windows), "room": len(rooms),
        },
        variation=variation,
    )


def compact_thin() -> SyntheticPlan:
    walls = [
        (120, 100, 880, 100), (880, 100, 880, 660),
        (880, 660, 120, 660), (120, 660, 120, 100),
        (500, 100, 500, 660), (120, 370, 880, 370),
    ]
    doors = [
        ((500, 210), 90, 0, (500, 210), (500, 270), 60),
        ((350, 370), 180, 270, (290, 370), (350, 370), 60),
        ((650, 370), 0, 90, (650, 370), (710, 370), 60),
    ]
    windows = [
        ((300, 100), 100, "H"), ((700, 100), 110, "H"),
        ((880, 520), 100, "V"),
    ]
    rooms = [
        (129, 109, 491, 361, "OFFICE"), (509, 109, 871, 361, "LIVING"),
        (129, 379, 491, 651, "BEDROOM"), (509, 379, 871, 651, "BATH"),
    ]
    return _render_plan(
        "compact_thin", (1000, 760), 16, walls, doors, windows, rooms,
        ("compact", "thin walls", "small door symbols", "low resolution"),
    )


def asymmetric_medium() -> SyntheticPlan:
    walls = [
        (180, 130, 1220, 130), (1220, 130, 1220, 850),
        (1220, 850, 180, 850), (180, 850, 180, 130),
        (650, 130, 650, 850), (180, 500, 650, 500),
        (650, 610, 1220, 610), (930, 610, 930, 850),
    ]
    doors = [
        ((650, 300), 90, 0, (650, 300), (650, 380), 80),
        ((430, 500), 180, 270, (350, 500), (430, 500), 80),
        ((650, 720), 90, 180, (650, 720), (650, 800), 80),
        ((930, 700), 90, 0, (930, 700), (930, 780), 80),
    ]
    windows = [
        ((330, 130), 150, "H"), ((850, 130), 180, "H"),
        ((1100, 850), 140, "H"), ((1220, 360), 150, "V"),
    ]
    rooms = [
        (192, 142, 638, 488, "OFFICE"),
        (662, 142, 1208, 598, "LIVING"),
        (192, 512, 638, 838, "BEDROOM"),
        (662, 622, 918, 838, "BATH"),
        (942, 622, 1208, 838, "LAUNDRY"),
    ]
    return _render_plan(
        "asymmetric_medium", (1400, 1000), 22, walls, doors, windows, rooms,
        ("asymmetric layout", "mixed room sizes", "medium resolution"),
    )


def dense_large() -> SyntheticPlan:
    walls = [
        (160, 120, 1640, 120), (1640, 120, 1640, 1160),
        (1640, 1160, 160, 1160), (160, 1160, 160, 120),
        (590, 120, 590, 1160), (1040, 120, 1040, 1160),
        (160, 470, 1040, 470), (590, 790, 1640, 790),
        (1320, 120, 1320, 470),
    ]
    doors = [
        ((590, 260), 90, 0, (590, 260), (590, 365), 105),
        ((400, 470), 180, 270, (295, 470), (400, 470), 105),
        ((820, 470), 0, 90, (820, 470), (925, 470), 105),
        ((1040, 610), 90, 180, (1040, 610), (1040, 715), 105),
        ((760, 790), 180, 270, (655, 790), (760, 790), 105),
        ((1320, 320), 90, 0, (1320, 320), (1320, 425), 105),
    ]
    windows = [
        ((350, 120), 190, "H"), ((800, 120), 220, "H"),
        ((1470, 120), 180, "H"), ((160, 650), 200, "V"),
        ((1640, 610), 220, "V"), ((1400, 1160), 200, "H"),
    ]
    rooms = [
        (176, 136, 574, 454, "OFFICE"), (606, 136, 1024, 454, "LIVING"),
        (1056, 136, 1304, 454, "BATH"), (1336, 136, 1624, 454, "STORAGE"),
        (176, 486, 574, 1144, "BEDROOM"), (606, 486, 1024, 774, "LAUNDRY"),
        (1056, 486, 1624, 774, "GYM"), (606, 806, 1624, 1144, "REC ROOM"),
    ]
    return _render_plan(
        "dense_large", (1800, 1300), 30, walls, doors, windows, rooms,
        ("dense layout", "thick walls", "large symbols", "high resolution"),
    )


def _rotate_plan(plan: SyntheticPlan, degrees: float, name: str) -> SyntheticPlan:
    height, width = plan.image.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    image = cv2.warpAffine(
        plan.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    truth = {
        key: cv2.warpAffine(
            mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        for key, mask in plan.truth.items()
    }
    return SyntheticPlan(
        name=name, image=image, truth=truth,
        truth_objects=dict(plan.truth_objects),
        variation=(*plan.variation, f"rotation {degrees:+.1f} degrees"),
    )


def skewed_medium() -> SyntheticPlan:
    return _rotate_plan(asymmetric_medium(), 1.5, "skewed_medium")


def degraded_dense() -> SyntheticPlan:
    plan = dense_large()
    small = cv2.resize(
        plan.image, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA,
    )
    image = cv2.resize(
        small, (plan.image.shape[1], plan.image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    image = cv2.GaussianBlur(image, (3, 3), 0.65)
    rng = np.random.default_rng(20260714)
    noise = rng.normal(0.0, 2.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return SyntheticPlan(
        name="degraded_dense", image=image, truth=plan.truth,
        truth_objects=dict(plan.truth_objects),
        variation=(*plan.variation, "downsampled scan", "blur", "deterministic noise"),
    )


CASE_FACTORIES: dict[str, Callable[[], SyntheticPlan]] = {
    "compact_thin": compact_thin,
    "asymmetric_medium": asymmetric_medium,
    "dense_large": dense_large,
    "skewed_medium": skewed_medium,
    "degraded_dense": degraded_dense,
}


def pixel_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    pred = prediction > 0
    target = truth > 0
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "iou": round(iou, 4),
    }


def summarize_cases(case_results: dict[str, dict]) -> dict:
    ranked = [
        (metrics["iou"], case_name, class_name)
        for case_name, result in case_results.items()
        for class_name, metrics in result["classes"].items()
    ]
    worst_iou, worst_case, worst_class = min(ranked)
    per_case_macro = {
        name: round(float(np.mean([
            metrics["iou"] for metrics in result["classes"].values()
        ])), 4)
        for name, result in case_results.items()
    }
    weakest_plan = min(per_case_macro, key=per_case_macro.get)
    return {
        "per_case_macro_iou": per_case_macro,
        "worst_floorplan": {
            "name": weakest_plan, "macro_iou": per_case_macro[weakest_plan],
        },
        "worst_class_case": {
            "floorplan": worst_case, "class": worst_class, "iou": worst_iou,
        },
    }


def _comparison_image(plan: SyntheticPlan, predictions: dict[str, np.ndarray]) -> np.ndarray:
    base = cv2.cvtColor(plan.image, cv2.COLOR_GRAY2BGR)
    height, width = base.shape[:2]
    canvas = np.full((height, width * 3, 3), 255, np.uint8)
    canvas[:, :width] = base
    truth_overlay = base.copy()
    pred_overlay = base.copy()
    for class_name in CLASSES:
        colour = np.asarray(CLASS_COLOURS[class_name], np.uint8)
        truth_owned = plan.truth[class_name] > 0
        pred_owned = predictions[class_name] > 0
        truth_overlay[truth_owned] = (
            0.45 * truth_overlay[truth_owned] + 0.55 * colour
        ).astype(np.uint8)
        pred_overlay[pred_owned] = (
            0.45 * pred_overlay[pred_owned] + 0.55 * colour
        ).astype(np.uint8)
    canvas[:, width:2 * width] = truth_overlay
    canvas[:, 2 * width:] = pred_overlay
    for index, title in enumerate(("SOURCE", "GROUND TRUTH", "PREDICTION")):
        cv2.putText(
            canvas, title, (index * width + 18, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return canvas


def evaluate_plan(plan: SyntheticPlan, output_dir: Path, ocr_workers: int) -> dict:
    case_dir = output_dir / plan.name
    case_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(case_dir / "source.png"), plan.image)
    for class_name, mask in plan.truth.items():
        cv2.imwrite(str(case_dir / f"truth_{class_name}.png"), mask)

    config = PipelineConfig(ocr_parallel_workers=max(1, ocr_workers))
    state = run_pipeline_state(image=plan.image, config=config)
    predictions = {
        "wall": state.wall_mask,
        "door": state.door_mask,
        "window": state.window_mask,
        "room": state.room_region_mask,
    }
    missing = [name for name, mask in predictions.items() if mask is None]
    if missing:
        raise RuntimeError(f"{plan.name}: pipeline did not export masks: {missing}")
    for class_name, mask in predictions.items():
        cv2.imwrite(str(case_dir / f"pred_{class_name}.png"), mask)
    cv2.imwrite(
        str(case_dir / "comparison.png"), _comparison_image(plan, predictions),
    )

    predicted_objects = {
        "wall": len(state.walls), "door": len(state.doors),
        "window": len(state.windows), "room": len(state.rooms),
    }
    classes = {
        name: {
            **pixel_metrics(predictions[name], plan.truth[name]),
            "truth_objects": plan.truth_objects[name],
            "pred_objects": predicted_objects[name],
            "object_count_error": predicted_objects[name] - plan.truth_objects[name],
        }
        for name in CLASSES
    }
    return {
        "variation": list(plan.variation),
        "image_size": [int(plan.image.shape[1]), int(plan.image.shape[0])],
        "classes": classes,
        "stage_timings_seconds": {
            key: round(value, 4) for key, value in state.debug.stage_timings.items()
        },
        "pipeline_counts": dict(state.debug.segment_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default=",".join(CASE_FACTORIES),
        help="comma-separated generated cases, or 'all'",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "evaluation_output" / "generalization",
    )
    parser.add_argument("--ocr-workers", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print("\n".join(CASE_FACTORIES))
        return 0
    names = list(CASE_FACTORIES) if args.cases == "all" else [
        name.strip() for name in args.cases.split(",") if name.strip()
    ]
    unknown = sorted(set(names) - set(CASE_FACTORIES))
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        logging.info("running validation case %s", name)
        results[name] = evaluate_plan(
            CASE_FACTORIES[name](), args.output_dir, args.ocr_workers,
        )
    report = {
        "cases": results,
        "summary": summarize_cases(results),
        "notes": [
            "Synthetic masks are exact generated labels, never pipeline predictions.",
            "ARCH-3 real-overlay metrics are evaluated separately with evaluate_annotation.py.",
            "ARCH-5 remains unscored until a matching ground truth is supplied.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    logging.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
