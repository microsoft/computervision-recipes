# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for visualizations
"""
from pathlib import Path
from typing import Dict, List, Union, Tuple, Callable, Any, Iterator, Optional

import numpy as np
import PIL
from PIL import Image, ImageDraw
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib

from .bbox import _Bbox, DetectionBbox
from .mask import binarise_mask, colorise_binary_mask, transparentise_mask
from .model import ims_eval_detections
from .references.coco_eval import CocoEvaluator
from ..common.misc import get_font


class PlotSettings:
    """ Simple class to contain bounding box params. """

    def __init__(
        self,
        rect_th: int = 4,
        rect_color: Tuple[int, int, int] = (0, 0, 255),
        text_size: int = 20,
        text_color: Tuple[int, int, int] = (0, 0, 255),
        mask_color: Tuple[int, int, int] = (0, 0, 128),
        mask_alpha: float = 0.8,
        keypoint_th: int = 3,
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
    ):
        self.rect_th = rect_th
        self.rect_color = rect_color
        self.text_size = text_size
        self.text_color = text_color
        self.mask_color = mask_color
        self.mask_alpha = mask_alpha
        self.keypoint_th = keypoint_th
        self.keypoint_color = keypoint_color

        # Create color table
        colors = matplotlib.cm.get_cmap("tab20").colors
        colors = np.floor(np.array(colors) * 255).astype("int")
        self._colors = tuple(map(tuple, colors))

    def get_colors(self, index):
        dark_color = self._colors[(index % 10) * 2]
        bright_color = self._colors[(index % 10) * 2 + 1]
        return dark_color, bright_color


def plot_boxes_stats(
    data, show: bool = True, figsize: tuple = (18, 3)
) -> None:
    """Plot statistics such as number of annotations for class, or
        distribution of width/height of the annotations.

    Args:
        data: detection dataset.
        show: Show plot. Use False if want to manually show the plot later.
        figsize: Figure size (w, h).
    """
    # Get annotation statistics
    labels_counts, box_widths, box_heights, box_rel_widths, box_rel_heights = (
        data.boxes_stats()
    )

    # Plot results
    plt.subplots(1, 3, figsize=figsize)
    plt.subplot(1, 3, 1)
    class_names = [l for [l, c] in labels_counts.most_common()][::-1]
    counts = [c for [l, c] in labels_counts.most_common()][::-1]
    plt.barh(range(len(class_names)), counts)
    plt.gca().set_yticks(range(len(class_names)))
    plt.gca().set_yticklabels(class_names)
    plt.xlabel("Number of annotations per class")

    plt.subplot(1, 3, 2)
    plt.hist([box_widths, box_heights], 20, label=["Width", "Height"])
    plt.xlabel("Distribution of box sizes")
    plt.legend()
    plt.ylabel("Pixels")

    plt.subplot(1, 3, 3)
    plt.hist(
        [box_rel_widths, box_rel_heights],
        20,
        label=["Normalized width", "Normalized height"],
    )
    plt.xlabel("Distribution of box sizes [normalized by image dimension]")
    plt.legend()
    plt.ylabel("Pixels")

    if show:
        plt.show()


# ===== Plotting of ground truth and prediction =====


def plot_boxes(
    im: PIL.Image.Image,
    bboxes: List[_Bbox],
    title: str = None,
    plot_settings: PlotSettings = PlotSettings(),
) -> PIL.Image.Image:
    """ Plot boxes on Image and return the Image

    Args:
        im: The image to plot boxes on
        bboxes: a list of bboxes (either DetectionBbox or AnnotationBbox)
        title: optional title str to pass in to draw on the top of the image
        plot_settings: the parameter of the bounding boxes

    Returns:
        The same image with boxes and labels plotted on it
    """
    if len(bboxes) > 0:
        draw = ImageDraw.Draw(im)
        font = get_font(size=plot_settings.text_size)

        for bbox in bboxes:
            # do not draw background bounding boxes
            if hasattr(bbox, "label_idx") and bbox.label_idx == 0:
                continue

            # show detection score in rectangle label
            bbox_text = bbox.label_name
            if type(bbox) is DetectionBbox:
                bbox_text += " ({:0.2f})".format(bbox.score)

            # pick rectangle and text color if set to None
            text_color = (
                plot_settings.text_color
                or plot_settings.get_colors(bbox.label_idx)[0]
            )
            rect_color = (
                plot_settings.rect_color
                or plot_settings.get_colors(bbox.label_idx)[1]
            )

            # draw rect
            box = [(bbox.left, bbox.top), (bbox.right, bbox.bottom)]
            draw.rectangle(
                box, outline=rect_color, width=plot_settings.rect_th
            )

            # write prediction class
            text_offset = plot_settings.text_size + plot_settings.rect_th
            draw.text(
                (bbox.left, max(0, bbox.top - text_offset)),
                bbox_text,
                font=font,
                fill=text_color,
            )

        if title is not None:
            draw.text((0, 0), title, font=font, fill=plot_settings.text_color)

    return im


def plot_masks(
    im: Union[str, Path, PIL.Image.Image],
    mask: Union[str, Path, np.ndarray],
    plot_settings: PlotSettings = PlotSettings(),
) -> PIL.Image.Image:
    """ Put mask onto image.

    Args:
        im: the image to plot masks on
        mask: it should be binary masks of [N, Height, Width], or grayscale
            mask of [Height, Width] with different values representing
            different objects, 0 as background
        plot_settings: the parameter to plot the masks
    """
    if isinstance(im, (str, Path)):
        im = Image.open(im)

    # convert to RGBA for transparentising
    im = im.convert("RGBA")
    # colorise masks
    binary_masks = binarise_mask(mask)
    colored_masks = [
        colorise_binary_mask(bmask, plot_settings.mask_color)
        for bmask in binary_masks
    ]
    # merge masks into img one by one
    for cmask in colored_masks:
        tmask = Image.fromarray(
            transparentise_mask(cmask, plot_settings.mask_alpha)
        )
        im = Image.alpha_composite(im, tmask)

    return im


def plot_keypoints(
    im: Union[str, Path, PIL.Image.Image],
    keypoints: np.ndarray,
    keypoint_meta: Dict = None,
    plot_settings: PlotSettings = PlotSettings(),
) -> PIL.Image.Image:
    """ Plot connected keypoints on Image and return the Image.

    Args:
        im: the image to plot keypoints on
        keypoints: the keypoints to plot, of shape (N, num_keypoints, 3),
            where N is the number of objects.  3 means x, y and visibility.
            0 for visibility means invisible
        keypoint_meta: meta data of keypoints which should include at least
            "skeleton"
        plot_settings: the parameter to plot the keypoints
    """
    if isinstance(im, (str, Path)):
        im = Image.open(im)

    if keypoints is not None:
        assert (
            keypoints.ndim == 3 and keypoints.shape[2] == 3
        ), "Malformed keypoints array"
        if keypoint_meta:
            assert (
                np.max(np.array(keypoint_meta["skeleton"]))
                < keypoints.shape[1]
            ), "Skeleton index out of range"

        draw = ImageDraw.Draw(im)

        # get connected skeleton lines of the keypoints
        if keypoint_meta:
            joints = keypoints[:, keypoint_meta["skeleton"]]
            visibles = (joints[..., 2] != 0).all(axis=2)
            bones = joints[visibles][..., :2]

            # draw skeleton lines
            for line in bones.reshape((-1, 4)).tolist():
                draw.line(
                    line,
                    fill=plot_settings.keypoint_color,
                    width=plot_settings.keypoint_th,
                )

        # draw keypoints
        visible_point_xys = keypoints[keypoints[..., 2] != 0][..., :2]
        offset = 2 * plot_settings.keypoint_th
        rects = np.hstack(
            [
                visible_point_xys - offset,  # left top
                visible_point_xys + offset,  # right bottom
            ]
        )
        for rect in rects.tolist():
            draw.ellipse(rect, fill=plot_settings.keypoint_color)

    return im


def plot_detections(
    detection: Dict,
    data: "DetectionDataset" = None,
    idx: int = None,
    keypoint_meta: Dict = None,
    ax: plt.axes = None,
    text_size: int = None,
    rect_th: int = None,
    keypoint_th = None,
) -> PIL.Image.Image:
    """ Put mask onto image.

    Args:
        detection: output running model prediction.
        data: dataset with ground truth information.
        idx: index into the data object to find the ground truth which corresponds to the detection.
        keypoint_meta: meta data of keypoints which should include at least
            "skeleton".
        ax: an optional ax to specify where you wish the figure to be drawn on
        text_size: override text size
        rect_th: override thickness of annotation rectangles
        key
    """
    # Open image
    assert detection["im_path"], 'Detection["im_path"] should not be None.'
    im = Image.open(detection["im_path"])

    default_plot_settings = PlotSettings()
    if not text_size: text_size = default_plot_settings.text_size
    if not rect_th: rect_th = default_plot_settings.rect_th
    if not keypoint_th: keypoint_th = default_plot_settings.keypoint_th

    # Adjust the rectangle thickness etc. to the image resolution
    scale = max(im.size) / 500.0
    rect_th = int(rect_th * scale)
    text_size = int(text_size * scale)
    keypoint_th = int(keypoint_th * scale)

    # Get id of ground truth image/annotation
    if data and idx is None:
        idx = detection["idx"]

    # Loop over all images
    det_bboxes = detection["det_bboxes"]

    # Plot ground truth mask
    if data and data.mask_paths:
        mask_path = data.mask_paths[idx]
        if mask_path:
            im = plot_masks(
                im,
                mask_path,
                plot_settings=PlotSettings(mask_color=(0, 128, 0)),
            )

    # Plot predicted masks
    if "masks" in detection:
        mask = detection["masks"]
        im = plot_masks(im, mask, PlotSettings(mask_color=(128, 165, 0)))

    # Plot ground truth keypoints
    if data and data.keypoints and data.keypoint_meta:
        im = plot_keypoints(
            im,
            data.keypoints[idx],
            data.keypoint_meta,
            PlotSettings(
                keypoint_color=(0, 192, 0),
                rect_th=rect_th,
                text_size=text_size,
                keypoint_th=keypoint_th,
            ),
        )

    # Plot predicted keypoints
    if "keypoints" in detection:
        im = plot_keypoints(
            im,
            detection["keypoints"],
            keypoint_meta,
            PlotSettings(
                keypoint_color=(192, 165, 0),
                rect_th=rect_th,
                text_size=text_size,
                keypoint_th=keypoint_th,
            ),
        )

    # Plot the detections
    plot_boxes(
        im,
        det_bboxes,
        plot_settings=PlotSettings(
            rect_color=None,
            text_color=None,
            rect_th=rect_th,
            text_size=text_size,
            keypoint_th=keypoint_th,
        ),
    )

    # Plot the ground truth annotations
    if data:
        anno_bboxes = data.anno_bboxes[idx]
        plot_boxes(
            im,
            anno_bboxes,
            plot_settings=PlotSettings(
                rect_color=(0, 255, 0),
                text_color=(0, 255, 0),
                rect_th=int(0.5 * rect_th),
                text_size=text_size,
                keypoint_th=keypoint_th,
            ),
        )

    # show image
    if ax:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im)
    else:
        return im


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
    figsize = (figsize[0], fig_height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 or cols == 1:
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


# ===== Precision - Recall curve =====


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
        rec_thrs: the recall thresholds to return
        cat_ids: label ids to use for evaluation
        area_rng: object area ranges for evaluation
        max_dets: thresholds on max detections per image

    Return the settings as a tuple to be passed into:
    `coco_eval.eval['precision']`
    """
    return iou_thrs, rec_thrs, cat_ids, area_rng, max_dets


def _plot_pr_curve_iou_range(
    ax: plt.axes, coco_eval: CocoEvaluator, iou_type: Optional[str] = None
) -> None:
    """ Plots the PR curve over varying iou thresholds averaging over [K]
    categories. """
    x = np.arange(0.0, 1.01, 0.01)
    iou_thrs_idx = range(0, 10)
    iou_thrs = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )

    # get_cmap() - a function that maps each index in 0, 1, ..., n-1 to a distinct
    # RGB color; the keyword argument name must be a standard mpl colormap name.
    cmap = plt.cm.get_cmap("hsv", len(iou_thrs))

    ax = _setup_pr_axes(
        ax, f"Precision-Recall Curve ({iou_type}) @ different IoU Thresholds"
    )
    for i, c in zip(iou_thrs_idx, iou_thrs):
        arr = coco_eval.eval["precision"][_get_precision_recall_settings(i)]
        arr = np.average(arr, axis=1)
        ax.plot(x, arr, c=cmap(i), label=f"IOU={round(c, 2)}")

    ax.legend(loc="lower left")


def _plot_pr_curve_iou_mean(
    ax: plt.axes, coco_eval: CocoEvaluator, iou_type: Optional[str] = None
) -> None:
    """ Plots the PR curve, averaging over iou thresholds and [K] labels. """
    x = np.arange(0.0, 1.01, 0.01)
    ax = _setup_pr_axes(
        ax, f"Precision-Recall Curve ({iou_type}) - Mean over IoU Thresholds"
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

    nrows = len(evaluator.coco_eval)
    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    for i, (k, coco_eval) in enumerate(evaluator.coco_eval.items()):
        _plot_pr_curve_iou_range(
            axes[i, 0] if nrows > 1 else axes[0], coco_eval, k
        )
        _plot_pr_curve_iou_mean(
            axes[i, 1] if nrows > 1 else axes[1], coco_eval, k
        )

    plt.show()


# ===== Correct/missing detection counts curve =====


def _plot_counts_curves_im(
    ax: plt.axes,
    score_thresholds: List[float],
    im_error_counts: List[int],
    im_wrong_det_counts: List[int],
    im_missed_gt_counts: List[int],
    im_neg_det_counts: List[int],
) -> None:
    """ Plot image-level correct/incorrect counts vs score thresholds """
    if im_neg_det_counts:
        ax.plot(
            score_thresholds,
            im_neg_det_counts,
            "y",
            label="Negative images with detections",
        )
    ax.plot(
        score_thresholds,
        im_error_counts,
        "r",
        label="Images with missed gt or wrong detections",
    )
    ax.plot(
        score_thresholds,
        im_wrong_det_counts,
        "g:",
        label="Images with wrong detections",
    )
    ax.plot(
        score_thresholds,
        im_missed_gt_counts,
        "b:",
        label="Images with missed ground truth",
    )

    ax.legend()
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Frequency")
    ax.set_title("Image counts", fontsize=14)
    ax.grid(True)


def _plot_counts_curves_obj(
    ax: plt.axes,
    score_thresholds: List[float],
    obj_missed_gt_counts: List[int],
    obj_wrong_det_counts: List[int],
    obj_neg_det_counts: List[int],
) -> None:
    """ Plot object-level correct/incorrect counts vs score thresholds """
    if obj_neg_det_counts:
        ax.plot(
            score_thresholds,
            obj_neg_det_counts,
            "y",
            label="Total number of detections within negative images",
        )
    ax.plot(
        score_thresholds,
        obj_wrong_det_counts,
        "g:",
        label="Total number of wrong detections",
    )
    ax.plot(
        score_thresholds,
        obj_missed_gt_counts,
        "b:",
        label="Total number of missed ground truths",
    )

    ax.legend()
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Frequency")
    ax.set_title("Object counts", fontsize=14)
    ax.grid(True)


def plot_counts_curves(
    detections: List[Dict],
    data_ds: Subset,
    detections_neg: List[Dict] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """ Plot object-level and image-level correct/incorrect counts vs score thresholds

    Args:
        detections: Detector prediction output for all test images
        data_ds: Test dataset, used to extract ground truth bboxes
        detections_neg: Detector prediction output for all negative images
        figsize: the figsize to plot the two graphs across

    Returns nothing, but plots count graphs.
    """
    # compute image and object level counts
    (
        score_thresholds,
        im_error_counts,
        im_wrong_det_counts,
        im_missed_gt_counts,
        obj_wrong_det_counts,
        obj_missed_gt_counts,
        im_neg_det_counts,
        obj_neg_det_counts,
    ) = ims_eval_detections(detections, data_ds, detections_neg)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _plot_counts_curves_im(
        ax1,
        score_thresholds,
        im_error_counts,
        im_wrong_det_counts,
        im_missed_gt_counts,
        im_neg_det_counts,
    )
    _plot_counts_curves_obj(
        ax2,
        score_thresholds,
        obj_missed_gt_counts,
        obj_wrong_det_counts,
        obj_neg_det_counts,
    )
    plt.show()
