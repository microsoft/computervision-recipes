# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for visualizations
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch import Tensor
from typing import Callable


def plot_thresholds(
    metric_function: Callable[[Tensor, Tensor, float], Tensor],
    y_pred: Tensor,
    y_true: Tensor,
    samples: int = 21,
    figsize: tuple = (12, 6),
) -> None:
    """ Plot the evaluation metric of the model at different thresholds.

    This function will plot the metric for every 0.05 increments of the
    threshold. This means that there will be a total of 20 increments.

    Args:
        metric_function: The metric function
        y_pred: predicted probabilities.
        y_true: True class indices.
        samples: Number of threshold samples
        figsize: Figure size (w, h)
    """
    metric_name = metric_function.__name__
    metrics = []
    for threshold in np.linspace(0, 1, samples):
        metric = metric_function(y_pred, y_true, threshold=threshold)
        metrics.append(metric)

    ax = pd.DataFrame(metrics).plot(figsize=figsize)
    ax.set_title(f"{metric_name} at different thresholds")
    ax.set_ylabel(f"{metric_name}")
    ax.set_xlabel("threshold")
    ax.set_xticks(np.linspace(0, 20, 11))
    ax.set_xticklabels(np.around(np.linspace(0, 1, 11), decimals=2))


def plot_pr_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: iter,
    show: bool = True,
    figsize: tuple = (12, 6),
):
    """Plot precision-recall and ROC curves .

    Currently, plots precision-recall and ROC curves.

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
        figsize (tuple): Figure size (w, h).
    """
    plt.subplots(2, 2, figsize=figsize)

    plt.subplot(1, 2, 1)
    plot_precision_recall_curve(y_true, y_score, classes, False)

    plt.subplot(1, 2, 2)
    plot_roc_curve(y_true, y_score, classes, False)

    if show:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, classes: iter, show: bool = True
):
    """Plot receiver operating characteristic (ROC) curves and ROC areas.

    If the given class labels are multi-label, it binarizes the classes and plots each ROC along with an averaged ROC.
    For the averaged ROC, micro-average is used.
    See details from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
    """
    assert (
        len(classes) == y_score.shape[1]
        if len(y_score.shape) == 2
        else len(classes) == 2
    )

    # Set random colors seed for reproducibility.
    np.random.seed(123)

    # Reference line
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    # Plot ROC curve
    if len(classes) == 2:
        # If y_score is soft-max output from a binary-class problem, we use the second node's output only.
        if len(y_score.shape) == 2:
            y_score = y_score[:, 1]
        _plot_roc_curve(y_true, y_score)
    else:
        y_true = label_binarize(y_true, classes=list(range(len(classes))))
        _plot_multi_roc_curve(y_true, y_score, classes)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower left")

    if show:
        plt.show()


def _plot_multi_roc_curve(y_true, y_score, classes):
    # Plot ROC for each class
    if len(classes) > 2:
        for i in range(len(classes)):
            _plot_roc_curve(y_true[:, i], y_score[:, i], classes[i])

    # Compute micro-average ROC curve and ROC area
    _plot_roc_curve(y_true.ravel(), y_score.ravel(), "avg")


def _plot_roc_curve(y_true, y_score, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if label == "avg":
        lw = 2
        prefix = "Averaged ROC"
    else:
        lw = 1
        prefix = "ROC" if label is None else f"ROC for {label}"

    plt.plot(
        fpr,
        tpr,
        color=_generate_color(),
        label=f"{prefix} (area = {roc_auc:0.2f})",
        lw=lw,
    )


def plot_precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray, classes: iter, show: bool = True
):
    """Plot precision-recall (PR) curves.

    If the given class labels are multi-label, it binarizes the classes and plots each PR along with an averaged PR.
    For the averaged PR, micro-average is used.
    See details from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
    """
    assert (
        len(classes) == y_score.shape[1]
        if len(y_score.shape) == 2
        else len(classes) == 2
    )

    # Set random colors seed for reproducibility.
    np.random.seed(123)

    # Plot ROC curve
    if len(classes) == 2:
        # If y_score is soft-max output from a binary-class problem, we use the second node's output only.
        if len(y_score.shape) == 2:
            y_score = y_score[:, 1]
        _plot_precision_recall_curve(
            y_true, y_score, average_precision_score(y_true, y_score)
        )
    else:
        y_true = label_binarize(y_true, classes=list(range(len(classes))))
        _plot_multi_precision_recall_curve(y_true, y_score, classes)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")

    if show:
        plt.show()


def _plot_multi_precision_recall_curve(y_true, y_score, classes):
    # Plot PR for each class
    if len(classes) > 2:
        for i in range(len(classes)):
            _plot_precision_recall_curve(
                y_true[:, i],
                y_score[:, i],
                average_precision_score(y_true[:, i], y_score[:, i]),
                classes[i],
            )

    # Plot averaged PR. A micro-average is used
    _plot_precision_recall_curve(
        y_true.ravel(),
        y_score.ravel(),
        average_precision_score(y_true, y_score, average="micro"),
        "avg",
    )


def _plot_precision_recall_curve(y_true, y_score, ap, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    if label == "avg":
        lw = 2
        prefix = "Averaged precision-recall"
    else:
        lw = 1
        prefix = (
            "Precision-recall"
            if label is None
            else f"Precision-recall for {label}"
        )

    plt.plot(
        recall,
        precision,
        color=_generate_color(),
        label=f"{prefix} (area = {ap:0.2f})",
        lw=lw,
    )


def _generate_color():
    return np.random.rand(3)
