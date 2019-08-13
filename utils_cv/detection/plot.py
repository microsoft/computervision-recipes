# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for visualizations
"""
from typing import List, Union, Tuple
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .references.coco_eval import CocoEvaluator


def display_bounding_boxes(
    boxes: List[List[int]],
    categories: List[str],
    im_path: Union[Path, str],
    ax: Union[None, plt.axes] = None,
    rect_th: int = 2,
    rect_color: Tuple[int, int, int] = (255, 0, 0),
    text_size: float = 1,
    text_th: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """ Draw image with bounding boxes.

    Args:
        boxes: A list of [xmin, ymin, xmax, ymax] bounding boxes to draw
        categories: A list of detected categories
        im_path: the location of image path to draw
        ax: an optional ax to specify where you wish the figure to be drawn on

    Returns nothing, but plots the image with bounding boxes and categories.
    """

    # Read image with cv2
    im = cv2.imread(str(im_path))

    # Convert to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if len(boxes) > 0:
        for box, category in zip(boxes, categories):

            # reformat boxes to be consumable by cv2
            box = [(box[0], box[1]), (box[2], box[3])]

            # Draw Rectangle with the coordinates
            cv2.rectangle(
                im, box[0], box[1], color=rect_color, thickness=rect_th
            )

            # Write the prediction class
            cv2.putText(
                im,
                category,
                (box[0][0], box[0][1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                color=text_color,
                thickness=text_th,
            )

    # display the output image
    if ax is not None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im)
    else:
        plt.figure(figsize=figsize)
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
        plt.show()


def _get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def _setup_pr_axes(ax: plt.axes, title: str) -> plt.axes:
    """ Setup the plot settings for plotting PR curves. """
    ax.set_xlabel("recall", fontsize=12)
    ax.set_ylabel("precision", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.01)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    return ax


def _get_precision_recalls_settings(
    iou_thrs: Union[int, slice],
    rec_thrs: Union[int, slice] = slice(0, 101),
    cat_ids: int = 0,
    area_rng: int = 0,
    max_dets: int = 2,
) -> Tuple[Union[int, slice], Union[int, slice], int, int, int]:
    """ Returns the indices or slices needed to index into the
    coco_eval.eval['precision'] object.

    coco_eval.eval['precision'] is a 5-dimensional array. Each dimension
    represents the following:
    1. [T] 10 evenly distributed thresholds for IoU, from 0.5 to 0.95.
    2. [R] 101 recall thresholds, from 0 to 101
    3. [K] category, set to 0 if you want to display the results of the first
    category
    4. [A] area size range of the target (all-0, small-1, medium-2, large-3)
    5. [M] The maximum number of detection frames in a single image where index
    0 represents max_det=1, 1 represents max_det=10, 2 represents max_det=100

    Therefore, coco_eval.eval['precision'][0, :, 0, 0, 2] represents the value
    of 101 precisions corresponding to 101 recalls from 0 to 100 when IoU=0.5.

    Args:
        iou_thrs: the IoU thresholds to return
        rec_thrs: the recall thrsholds to return
        cat_ids: category ids to use for evaluation
        area_rng: object area ranges for evaluation
        max_dets: thresholds on max detections per image

    Return the settings as a tuple to be passed into:
    `coco_eval.eval['precision']`
    """
    return (iou_thrs, rec_thrs, cat_ids, area_rng, max_dets)


def _plot_pr_curve_iou_range(ax: plt.axes, coco_eval: CocoEvaluator) -> None:
    """ Plots the PR curve over varying iou thresholds. """
    x = np.arange(0.0, 1.01, 0.01)
    iou_thrs_idx = range(0, 10)
    iou_thrs = np.linspace(
        0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
    )
    cmap = _get_cmap(len(iou_thrs))

    ax = _setup_pr_axes(
        ax, "Precision-Recall Curve @ different IoU Thresholds"
    )
    for i, c in zip(iou_thrs_idx, iou_thrs):
        arr = coco_eval.eval["precision"][_get_precision_recalls_settings(i)]
        ax.plot(x, arr, c=cmap(i), label=f"IOU={round(c, 2)}")

    ax.legend(loc="lower left")


def _plot_pr_curve_iou_mean(ax: plt.axes, coco_eval: CocoEvaluator) -> None:
    """ Plots the PR curve, averaging the iou thresholds. """
    x = np.arange(0.0, 1.01, 0.01)
    ax = _setup_pr_axes(
        ax, "Precision-Recall Curve - Mean over IoU Thresholds"
    )
    avg_arr = np.mean(
        coco_eval.eval["precision"][
            _get_precision_recalls_settings(slice(0, None))
        ],
        axis=0,
    )

    ax.plot(x, avg_arr, c="black", label=f"IOU=mean")
    ax.legend(loc="lower left")


def plot_pr_curves(
    evaluator: CocoEvaluator, figsize: Tuple[int, int] = (16, 8)
) -> None:
    """ Plots two graphs to illustrate the Precision Recall.

    Args:
        evaluator: CocoEvaluator to get evaluation results from
        figsize: the figsize to plot the two graphs across

    Raises:
        Exception if accumulate hasn't been called on the passed in
        CocoEvaluator

    Returns nothing, but plots PR graphs.
    """
    for _, coco_eval in evaluator.coco_eval.items():
        if not coco_eval.eval:
            raise Exception(
                "`accumulate()` has not been called on the passed in coco_eval object."
            )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        _plot_pr_curve_iou_range(ax1, coco_eval)
        _plot_pr_curve_iou_mean(ax2, coco_eval)
        plt.show()
