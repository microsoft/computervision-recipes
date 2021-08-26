import logging
from typing import Dict, List

import torch
from torchmetrics import F1, Accuracy, IoU, MetricCollection, Precision, Recall


def get_semantic_segmentation_metrics(
    num_classes: int, thresholds: List[float] = [0.5, 0.3]
) -> MetricCollection:
    """Construct MetricCollection of Segmentation Metrics

    Parameters
    ----------
    num_classes : int
        Number of classes
    thresholds : List[float]
        List of thresholds for different IOU computing

    Returns
    -------
    metrics : torchmetrics.MetricCollection
        Collection of Segmentation metrics
    """
    metrics = {
        "mean_accuracy": Accuracy(num_classes=num_classes, mdmc_average="global"),
        "per_class_accuracy": Accuracy(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_precision": Precision(num_classes=num_classes, mdmc_average="global"),
        "per_class_precision": Precision(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_recall": Recall(num_classes=num_classes, mdmc_average="global"),
        "per_class_recall": Recall(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_f1": F1(num_classes=num_classes, mdmc_average="global"),
        "per_class_f1": F1(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
    }

    for threshold in thresholds:
        threshold_string = str(threshold).replace(".", "_")
        metrics[f"mean_iou_{threshold_string}"] = IoU(
            num_classes=num_classes, reduction="elementwise_mean", threshold=threshold
        )
        metrics[f"per_class_iou_{threshold_string}"] = IoU(
            num_classes=num_classes, reduction="none", threshold=threshold
        )

    print(metrics)

    return MetricCollection(metrics)


def log_metrics(results: Dict[str, torch.Tensor], classes: List[str], split: str):
    """Log metrics to stdout and AML

    Parameters
    ----------
    results : Dict
        Key is the name of the metric, value is a metric tensor
        If the metric is a mean, it is a 0-dim tensor
        If the metric is per class, it is a C-dim tensor (C for number of classes)
    split : {"train", "val", "test"}
        Split that the metrics are for
    """
    # Import does not appear to work on some non-AML environments
    from azureml.core.run import Run, _SubmittedRun

    # Get script logger
    log = logging.getLogger(__name__)

    # Get AML context
    run = Run.get_context()

    split = split.capitalize()

    for metric_name, result in results.items():
        log_name = f"[{split}] {metric_name}"
        if "mean" in metric_name:
            result = float(result)

            if isinstance(run, _SubmittedRun):
                run.parent.log(log_name, result)
            run.log(log_name, result)
        elif "per_class" in metric_name:
            result = {c: float(r) for c, r in zip(classes, result)}

            # Steps are children the experiment they belong to so the parent
            # also needs a log
            if isinstance(run, _SubmittedRun):
                run.parent.log_row(log_name, **result)
            run.log_row(log_name, **result)

        log.info(f"{log_name}: {result}")
