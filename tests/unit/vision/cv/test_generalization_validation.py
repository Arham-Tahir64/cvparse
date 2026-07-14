import numpy as np

from tools.validate_generalization import (
    CASE_FACTORIES,
    CLASSES,
    asymmetric_medium,
    pixel_metrics,
    summarize_cases,
)


def test_generated_validation_cases_are_deterministic_and_fully_labelled():
    for name, factory in CASE_FACTORIES.items():
        first = factory()
        second = factory()
        assert first.name == name
        assert np.array_equal(first.image, second.image)
        assert set(first.truth) == set(CLASSES)
        assert first.image.ndim == 2
        for class_name in CLASSES:
            assert first.truth[class_name].shape == first.image.shape
            assert np.count_nonzero(first.truth[class_name]) > 0
            assert first.truth_objects[class_name] > 0


def test_generated_plan_contains_drafting_clutter_outside_truth_masks():
    plan = asymmetric_medium()
    foreground = plan.image < 128
    labelled = np.logical_or.reduce([mask > 0 for mask in plan.truth.values()])
    assert np.count_nonzero(foreground & ~labelled) > 0


def test_pixel_metrics_report_exact_and_disjoint_masks():
    truth = np.zeros((20, 20), np.uint8)
    truth[2:8, 2:8] = 255
    assert pixel_metrics(truth, truth) == {
        "tp": 36, "fp": 0, "fn": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0,
    }
    prediction = np.zeros_like(truth)
    prediction[12:18, 12:18] = 255
    metrics = pixel_metrics(prediction, truth)
    assert metrics["tp"] == 0
    assert metrics["fp"] == 36
    assert metrics["fn"] == 36
    assert metrics["iou"] == 0.0


def test_summary_exposes_worst_case_and_class_not_only_average():
    results = {
        "strong": {"classes": {
            name: {"iou": 0.9} for name in CLASSES
        }},
        "weak": {"classes": {
            "wall": {"iou": 0.8}, "door": {"iou": 0.1},
            "window": {"iou": 0.7}, "room": {"iou": 0.8},
        }},
    }
    summary = summarize_cases(results)
    assert summary["worst_floorplan"]["name"] == "weak"
    assert summary["worst_class_case"] == {
        "floorplan": "weak", "class": "door", "iou": 0.1,
    }
