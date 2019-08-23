# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for visualizations
"""
import os
from typing import List, Union, Tuple, Callable, Any, Iterator
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt

from .references.coco_eval import CocoEvaluator
from .bbox import AnnotationBbox


class PlotSettings:
    """ Simple class to contain bounding box params. """

    def __init__(
        self,
        rect_th: int = 4,
        rect_color: Tuple[int, int, int] = (255, 0, 0),
        text_size: int = 25,
        text_font: str = "DejaVuSerifCondensed.ttf",
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.rect_th, self.rect_color, self.text_size, self.text_font, self.text_color = (
            rect_th,
            rect_color,
            text_size,
            text_font,
            text_color,
        )


def plot_boxes(
    im: PIL.Image.Image,
    anno_bboxes: List[AnnotationBbox],
    im_title: str = None,
    plot_settings: PlotSettings = PlotSettings(),
) -> PIL.Image.Image:
    """ Plot boxes on Image and return the Image

    Args:
        im: The image to plot boxes on
        anno_bboxes: a list of annotations bboxes
        im_title: optional title str to pass in to draw on the top of the image
        plot_settings: the parameter of the bounding boxes

    Returns:
        The same image with boxes and labels plotted on it
    """
    if len(anno_bboxes) > 0:
        draw = ImageDraw.Draw(im)
        for anno_bbox in anno_bboxes:

            box = [
                (anno_bbox.left, anno_bbox.top),
                (anno_bbox.right, anno_bbox.bottom),
            ]

            # draw rect
            draw.rectangle(
                box,
                outline=plot_settings.rect_color,
                width=plot_settings.rect_th,
            )

            font = ImageFont.truetype(
                plot_settings.text_font, plot_settings.text_size
            )

            # write prediction class
            draw.text(
                (anno_bbox.left, anno_bbox.top),
                anno_bbox.label_name,
                font=font,
                fill=plot_settings.text_color,
            )

        if im_title is not None:
            draw.text(
                (0, 0), im_title, font=font, fill=plot_settings.text_color
            )

    return im


def display_bounding_boxes(
    anno_bboxes: List[AnnotationBbox],
    im_path: Union[Path, str],
    ax: Union[None, plt.axes] = None,
    plot_settings: PlotSettings = PlotSettings(),
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """ Draw image with bounding boxes.

    Args:
        anno_bboxes: A list of AnnotationsBboxes
        im_path: the location of image path to draw
        ax: an optional ax to specify where you wish the figure to be drawn on

    Returns nothing, but plots the image with bounding boxes and labels.
    """
    # Read image
    im = Image.open(str(im_path))

    # set an image title
    im_title = os.path.basename(im_path)

    # plot boxes on im
    im = plot_boxes(
        im, anno_bboxes, im_title=im_title, plot_settings=plot_settings
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


def plot_grid(
    plot_func: Callable[..., None],
    args: Union[Callable, Iterator, Any],
    rows: int = 1,
    cols: int = 3,
    figsize: Tuple[int, int] = (16, 16),
) -> None:
    """ Helper function to plot image grids.

    Args:
        plot_func: callback to call on each subplot. It should take an 'ax' as
        the last param.
        args: args can be passed in in many forms. It can be an iterator, a
        callable, or simply some static parameters. If it is an iterator, this
        function will call `next` on it each time. If it is a callable, this
        function will call the function and use the returned values each time.
        rows: rows to plot
        cols: cols to plot, default is 3. NOTE: use cols=3 for best looking
        grid
        figsize: figure size (will be dynamically modified in the code

    Returns nothing but plots graph
    """
    fig_height = rows * 8
    figsize = (16, fig_height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = [axes]

    for row in axes:
        for ax in row:
            # dynamic injection of params into callable
            arguments = (
                args()
                if isinstance(args, Callable)
                else (next(args) if hasattr(args, "__iter__") else args)
            )
            try:
                plot_func(arguments, ax)
            except Exception:
                plot_func(*arguments, ax)

    plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.1, wspace=0.2)


def plot_detection_vs_ground_truth(
    im_path: str,
    detection_annos: List[AnnotationBbox],
    ground_truth_annos: List[AnnotationBbox],
    ax: plt.axes,
) -> None:
    """ Plots bounding boxes of ground_truths and detections.

    Args:
        im_path: the image to plot
        detection_annos: a list of detected annotations
        ground_truth_annos: a list of ground_truth detections
        ax: the axis to plot on

    Returns nothing, but displays a graph
    """
    im = Image.open(im_path).convert("RGB")

    # plot detections
    pred_params = PlotSettings(rect_color=(255, 0, 0), text_size=1)
    im = plot_boxes(
        im,
        detection_annos,
        im_title=os.path.basename(im_path),
        plot_settings=pred_params,
    )

    # plot ground truth boxes
    ground_truth_params = PlotSettings(rect_color=(0, 255, 0), text_size=1)
    im = plot_boxes(
        im,
        ground_truth_annos,
        im_title=os.path.basename(im_path),
        plot_settings=ground_truth_params,
    )

    # show image
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im)


# ===== Precision - Recall curve =====


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


def _get_precision_recall_settings(
    iou_thrs: Union[int, slice],
    rec_thrs: Union[int, slice] = slice(0, None),
    cat_ids: int = slice(0, None),
    area_rng: int = 0,
    max_dets: int = 2,
) -> Tuple[Union[int, slice], Union[int, slice], int, int, int]:
    """ Returns the indices or slices needed to index into the
    coco_eval.eval['precision'] object.

    coco_eval.eval['precision'] is a 5-dimensional array. Each dimension
    represents the following:
    1. [T] 10 evenly distributed thresholds for IoU, from 0.5 to 0.95.
    2. [R] 101 recall thresholds, from 0 to 101
    3. [K] label, set to slice(0, None) to get precision over all the labels in
    the dataset. Then take the mean over all labels.
    4. [A] area size range of the target (all-0, small-1, medium-2, large-3)
    5. [M] The maximum number of detection frames in a single image where index
    0 represents max_det=1, 1 represents max_det=10, 2 represents max_det=100

    Therefore, coco_eval.eval['precision'][0, :, 0, 0, 2] represents the value
    of 101 precisions corresponding to 101 recalls from 0 to 100 when IoU=0.5.

    Args:
        iou_thrs: the IoU thresholds to return
        rec_thrs: the recall thrsholds to return
        cat_ids: label ids to use for evaluation
        area_rng: object area ranges for evaluation
        max_dets: thresholds on max detections per image

    Return the settings as a tuple to be passed into:
    `coco_eval.eval['precision']`
    """
    return (iou_thrs, rec_thrs, cat_ids, area_rng, max_dets)


def _plot_pr_curve_iou_range(ax: plt.axes, coco_eval: CocoEvaluator) -> None:
    """ Plots the PR curve over varying iou thresholds averaging over [K]
    categoyies. """
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
        arr = coco_eval.eval["precision"][_get_precision_recall_settings(i)]
        arr = np.average(arr, axis=1)
        ax.plot(x, arr, c=cmap(i), label=f"IOU={round(c, 2)}")

    ax.legend(loc="lower left")


def _plot_pr_curve_iou_mean(ax: plt.axes, coco_eval: CocoEvaluator) -> None:
    """ Plots the PR curve, averaging over iou thresholds and [K] labels. """
    x = np.arange(0.0, 1.01, 0.01)
    ax = _setup_pr_axes(
        ax, "Precision-Recall Curve - Mean over IoU Thresholds"
    )
    avg_arr = np.mean(  # mean over K labels
        np.mean(  # mean over iou thresholds
            coco_eval.eval["precision"][
                _get_precision_recall_settings(slice(0, None))
            ],
            axis=0,
        ),
        axis=1,
    )

    ax.plot(x, avg_arr, c="black", label=f"IOU=mean")
    ax.legend(loc="lower left")


def plot_pr_curves(
    evaluator: CocoEvaluator, figsize: Tuple[int, int] = (16, 8)
) -> None:
    """ Plots two graphs to illustrate the Precision Recall.

    This method uses the CocoEvaluator object from the references provided by
    pytorch, which in turn uses the COCOEval from pycocotools.

    source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        evaluator: CocoEvaluator to get evaluation results from
        figsize: the figsize to plot the two graphs across

    Raises:
        Exception if accumulate hasn't been called on the passed in
        CocoEvaluator

    Returns nothing, but plots PR graphs.
    """
    coco_eval = evaluator.coco_eval["bbox"]
    if not coco_eval.eval:
        raise Exception(
            "`accumulate()` has not been called on the passed in coco_eval object."
        )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _plot_pr_curve_iou_range(ax1, coco_eval)
    _plot_pr_curve_iou_mean(ax2, coco_eval)
    plt.show()
