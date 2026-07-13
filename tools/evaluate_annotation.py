"""Compare a generated debug annotation with the coloured reference overlay.

The evaluator reconstructs the unannotated source page at the reference
resolution, then detects overlays by testing whether each annotated pixel is
better explained by an alpha blend with a known semantic colour than by the
source pixel.  This avoids treating the floor-plan's black linework as an
annotation.  Metrics are calculated only in the largest connected annotated
region, excluding the legend and title block.

Usage:
    python tools/evaluate_annotation.py SOURCE.pdf goal.png generated.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np


# RGB colours are the two annotation schemes, not image-specific geometry.
# Multiple reference colours represent the distinct labelled room classes.
REFERENCE_STYLES = {
    "wall": [((142, 65, 196), 0.72)],
    "door": [((255, 92, 99), 0.62)],
    "window": [((91, 126, 244), 0.62)],
    "room": [
        ((255, 222, 91), 0.42),   # guest suite
        ((91, 222, 232), 0.35),  # bath
        ((164, 119, 238), 0.38), # gym/yoga
        ((100, 211, 180), 0.35), # laundry
        ((238, 166, 83), 0.38),  # linen
        ((255, 88, 111), 0.35),  # mechanical
        ((246, 145, 60), 0.38),  # storage
        ((190, 190, 190), 0.40), # circulation
        ((135, 207, 104), 0.38), # recreation
    ],
}

GENERATED_STYLES = {
    "wall": [((214, 38, 41), 0.85), ((255, 128, 13), 0.85)],
    "door": [((43, 161, 43), 1.0), ((43, 161, 43), 0.24)],
    "window": [((31, 119, 180), 0.65)],
    "room": [
        ((173, 199, 232), 0.25),
        ((255, 222, 91), 0.42),
        ((91, 222, 232), 0.35),
        ((164, 119, 238), 0.38),
        ((100, 211, 180), 0.35),
        ((238, 166, 83), 0.38),
        ((255, 88, 111), 0.35),
        ((246, 145, 60), 0.38),
        ((190, 190, 190), 0.40),
        ((135, 207, 104), 0.38),
    ],
}


def _render_source(pdf_path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    doc = fitz.open(pdf_path)
    page = doc[0]
    matrix = fitz.Matrix(width / page.rect.width, height / page.rect.height)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    if image.shape[:2] != shape:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _semantic_masks(
    annotated: np.ndarray,
    source: np.ndarray,
    styles: dict[str, list[tuple[tuple[int, int, int], float]]],
    improvement_threshold: float = 8.0,
) -> dict[str, np.ndarray]:
    ann = annotated.astype(np.float32)
    src = source.astype(np.float32)
    unchanged = np.linalg.norm(ann - src, axis=2)
    scores = []
    names = list(styles)
    for name in names:
        candidates = []
        for rgb, alpha in styles[name]:
            colour = np.asarray(rgb, np.float32)
            expected = src * (1.0 - alpha) + colour * alpha
            candidates.append(np.linalg.norm(ann - expected, axis=2))
        scores.append(np.min(candidates, axis=0))
    score_stack = np.stack(scores)
    winner = np.argmin(score_stack, axis=0)
    best = np.min(score_stack, axis=0)
    explained = unchanged - best > improvement_threshold
    return {name: explained & (winner == i) for i, name in enumerate(names)}


def _evaluation_roi(reference_masks: dict[str, np.ndarray]) -> np.ndarray:
    union = np.logical_or.reduce(list(reference_masks.values())).astype(np.uint8)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union, 8)
    if count <= 1:
        return np.ones_like(union, dtype=bool)
    # Room fills and wall bands form the largest annotated component.
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[label, :4]
    pad = max(5, round(max(w, h) * 0.025))
    roi = np.zeros_like(union, dtype=bool)
    roi[max(0, y - pad):min(roi.shape[0], y + h + pad),
        max(0, x - pad):min(roi.shape[1], x + w + pad)] = True
    return roi


def _boundary(mask: np.ndarray) -> np.ndarray:
    u8 = mask.astype(np.uint8)
    return cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    truth_boundary = _boundary(truth)
    pred_boundary = _boundary(pred)
    tolerance = np.ones((5, 5), np.uint8)
    truth_near = cv2.dilate(truth_boundary.astype(np.uint8), tolerance) > 0
    pred_near = cv2.dilate(pred_boundary.astype(np.uint8), tolerance) > 0
    b_precision = (np.count_nonzero(pred_boundary & truth_near) /
                   max(1, np.count_nonzero(pred_boundary)))
    b_recall = (np.count_nonzero(truth_boundary & pred_near) /
                max(1, np.count_nonzero(truth_boundary)))
    b_f1 = (2 * b_precision * b_recall / (b_precision + b_recall)
            if b_precision + b_recall else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "iou": round(iou, 4),
        "pixel_accuracy": round(accuracy, 4), "boundary_f1": round(b_f1, 4),
    }


def _component_count(mask: np.ndarray, min_area: int) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(int(area) >= min_area for area in stats[1:, cv2.CC_STAT_AREA])


def evaluate(source_pdf: Path, reference_path: Path, generated_path: Path) -> dict:
    reference = cv2.cvtColor(cv2.imread(str(reference_path)), cv2.COLOR_BGR2RGB)
    generated = cv2.cvtColor(cv2.imread(str(generated_path)), cv2.COLOR_BGR2RGB)
    h, w = reference.shape[:2]
    generated = cv2.resize(generated, (w, h), interpolation=cv2.INTER_AREA)
    source = _render_source(source_pdf, (h, w))
    reference_masks = _semantic_masks(reference, source, REFERENCE_STYLES)
    generated_masks = _semantic_masks(generated, source, GENERATED_STYLES)
    roi = _evaluation_roi(reference_masks)

    report = {"image_size": [w, h], "classes": {}}
    for name in REFERENCE_STYLES:
        truth = reference_masks[name] & roi
        pred = generated_masks[name] & roi
        values = _metrics(pred[roi], truth[roi])
        min_area = 4 if name in {"door", "window"} else 20
        values["truth_objects"] = _component_count(truth, min_area)
        values["pred_objects"] = _component_count(pred, min_area)
        values["truth_pixels"] = int(np.count_nonzero(truth))
        values["pred_pixels"] = int(np.count_nonzero(pred))
        report["classes"][name] = values

    truth_all = np.logical_or.reduce([m & roi for m in reference_masks.values()])
    pred_all = np.logical_or.reduce([m & roi for m in generated_masks.values()])
    report["overall_foreground"] = _metrics(pred_all[roi], truth_all[roi])
    report["macro_iou"] = round(float(np.mean([
        values["iou"] for values in report["classes"].values()
    ])), 4)
    report["macro_f1"] = round(float(np.mean([
        values["f1"] for values in report["classes"].values()
    ])), 4)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.source_pdf, args.reference, args.generated)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
