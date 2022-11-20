from typing import Dict

import pytest
import torch

from src.metrics.metrics import get_semantic_segmentation_metrics


def get_semantic_segmentation_metrics_cases():
    num_classes = 2

    pred_all_correct = torch.Tensor(
        [
            [
                [
                    [0.25, 0.25, 0.75, 0.75],
                    [0.25, 0.25, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                ],
                [
                    [0.75, 0.75, 0.25, 0.25],
                    [0.75, 0.75, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ]
        ]
    )

    ground_truth_all_correct = torch.Tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    ).int()

    expected_results_all_correct = {
        "mean_accuracy": torch.tensor(1),
        "per_class_accuracy": torch.Tensor([1, 1]),
        "mean_precision": torch.tensor(1),
        "per_class_precision": torch.Tensor([1, 1]),
        "mean_recall": torch.tensor(1),
        "per_class_recall": torch.Tensor([1, 1]),
        "mean_f1": torch.tensor(1),
        "per_class_f1": torch.Tensor([1, 1]),
        "mean_iou_0_5": torch.tensor(1),
        "per_class_iou_0_5": torch.Tensor([1, 1]),
        "mean_iou_0_3": torch.tensor(1),
        "per_class_iou_0_3": torch.Tensor([1, 1]),
    }

    all_correct_case = (
        pred_all_correct,
        ground_truth_all_correct,
        num_classes,
        expected_results_all_correct,
    )

    pred_none_correct = torch.Tensor(
        [
            [
                [
                    [0.25, 0.25, 0.75, 0.75],
                    [0.25, 0.25, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                ],
                [
                    [0.75, 0.75, 0.25, 0.25],
                    [0.75, 0.75, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ]
        ]
    )

    ground_truth_none_correct = torch.Tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ]
    ).int()

    expected_results_none_correct = {
        "mean_accuracy": torch.tensor(0),
        "per_class_accuracy": torch.Tensor([0, 0]),
        "mean_precision": torch.tensor(0),
        "per_class_precision": torch.Tensor([0, 0]),
        "mean_recall": torch.tensor(0),
        "per_class_recall": torch.Tensor([0, 0]),
        "mean_f1": torch.tensor(0),
        "per_class_f1": torch.Tensor([0, 0]),
        "mean_iou_0_5": torch.tensor(0),
        "per_class_iou_0_5": torch.Tensor([0, 0]),
        "mean_iou_0_3": torch.tensor(0),
        "per_class_iou_0_3": torch.Tensor([0, 0]),
    }

    none_correct_case = (
        pred_none_correct,
        ground_truth_none_correct,
        num_classes,
        expected_results_none_correct,
    )

    pred_typical_case = torch.Tensor(
        [
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.75, 0.75, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                ],
                [
                    [0.75, 0.75, 0.75, 0.75],
                    [0.75, 0.75, 0.75, 0.75],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ]
        ]
    )

    ground_truth_typical_case = torch.Tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    ).int()

    expected_results_typical_case = {
        "mean_accuracy": torch.tensor(0.75),
        "per_class_accuracy": torch.Tensor([2 / 3, 1]),
        "mean_precision": torch.tensor(0.75),
        "per_class_precision": torch.Tensor([1, 0.5]),
        "mean_recall": torch.tensor(0.75),
        "per_class_recall": torch.Tensor([2 / 3, 1]),
        "mean_f1": torch.tensor(0.75),
        "per_class_f1": torch.Tensor([0.8, 2 / 3]),
        "mean_iou_0_5": torch.tensor((2 / 3 + 1 / 2) / 2),
        "per_class_iou_0_5": torch.Tensor([2 / 3, 0.5]),
        "mean_iou_0_3": torch.tensor((2 / 3 + 1 / 2) / 2),
        "per_class_iou_0_3": torch.Tensor([2 / 3, 0.5]),
    }

    typical_case = (
        pred_typical_case,
        ground_truth_typical_case,
        num_classes,
        expected_results_typical_case,
    )

    cases = [all_correct_case, none_correct_case, typical_case]
    return cases


@pytest.mark.parametrize(
    "preds, ground_truth, num_classes, expected_results",
    get_semantic_segmentation_metrics_cases(),
)
def test_get_semantic_segmentation_metrics(
    preds: torch.Tensor,
    ground_truth: torch.Tensor,
    num_classes: int,
    expected_results: Dict,
):
    metrics = get_semantic_segmentation_metrics(num_classes, thresholds=[0.5, 0.3])
    metrics(preds, ground_truth)
    results = metrics.compute()

    assert len(results) > 0

    for metric_name, result in results.items():
        if "per_class" in metric_name:
            assert len(result) == num_classes
        assert torch.allclose(result, expected_results[metric_name].float())
