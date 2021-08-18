# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
from typing import List, Tuple, Union

from fastai.vision import pil2tensor, show_image
from fastai.vision.data import ImageDataBunch
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .dataset import load_im, load_mask, mask_area_sizes


# Plot original image(left), ground truth (middle), and overlaid ground truth (right)
def plot_image_and_mask(
    im_or_path: Union[np.ndarray, Union[str, Path]],
    mask_or_path: Union[np.ndarray, Union[str, Path]],
    show: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    alpha=0.50,
    cmap: ListedColormap = cm.get_cmap("Set3"),
) -> None:
    """ Plot an image and its ground truth mask.

    Args:
        im_or_path: image or path to image
        mask_or_path: mask or path to mask
        show: set to true to call matplotlib's show()
        figsize: figure size
        alpha: strength of overlying image on mask.
        cmap: mask color map.
    """
    im = load_im(im_or_path)
    mask = load_mask(mask_or_path)

    # Plot the image, the mask, and the mask overlaid on image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    show_image(im, ax=ax1)
    show_image(mask, ax=ax2, cmap=cmap)
    im.show(y=mask, ax=ax3, cmap=cmap, alpha=alpha)
    ax1.set_title("Image")
    ax2.set_title("Mask")
    ax3.set_title("Mask (overlaid on Image)")

    if show:
        plt.show()


def plot_segmentation(
    im_or_path: Union[np.ndarray, Union[str, Path]],
    pred_mask: Union[np.ndarray, Union[str, Path]],
    pred_scores: np.ndarray,
    gt_mask_or_path: Union[np.ndarray, Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 4),
    cmap: ListedColormap = cm.get_cmap("Set3"),
    ignore_background_label = True
) -> None:
    """ Plot an image, its predicted mask with associated scores, and optionally the ground truth mask.

    Args:
        im_or_path: image or path to image
        pred_mask: predicted mask
        pred_scores: pixel-wise confidence scores in the predictions
        gt_mask_or_path: ground truth mask or path to mask
        show: set to true to call matplotlib's show()
        figsize: figure size
        cmap: mask color map.
        ignore_background_label: set to True to ignore the 0 label.
    """
    im = load_im(im_or_path)
    pred_mask = pil2tensor(pred_mask, np.float32)
    if ignore_background_label:
        start_label = 1
    else:
        start_label = 0
    max_scores = np.max(np.array(pred_scores[start_label:]), axis=0)
    max_scores = pil2tensor(max_scores, np.float32)

    # Plot groud truth mask if provided
    if gt_mask_or_path:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        gt_mask = load_mask(gt_mask_or_path)
        show_image(gt_mask, ax=ax4, cmap=cmap)
        ax4.set_title("Ground truth mask")
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Plot image, predicted mask, and prediction scores
    show_image(im, ax=ax1)
    show_image(pred_mask, ax=ax2, cmap=cmap)
    show_image(max_scores, ax=ax3, cmap=cm.get_cmap("gist_heat"))
    ax1.set_title("Image")
    ax2.set_title("Predicted mask")
    ax3.set_title("Predicted scores")

    if show:
        plt.show()


def plot_mask_stats(
    data: ImageDataBunch,
    classes: List[str],
    show: bool = True,
    figsize: Tuple[int, int] = (15, 3),
    nr_bins: int = 50,
    exclude_classes: list = None,
) -> None:
    """ Plot statistics of the ground truth masks such as number or size of segments.

    Args:
        data: databunch with images and ground truth masks
        classes: list of class names
        show: set to true to call matplotlib's show()
        figsize: figure size
        nr_bins: number of bins for segment sizes histogram
        exclude_classes: list of classes to ignore, e.g. ["background"]
    """
    areas, pixel_counts = mask_area_sizes(data)
    class_names = [classes[k] for k,v in areas.items()]
    values_list = [v for k,v in areas.items()]
    seg_counts = [len(v) for v in values_list]
    pixel_counts = [np.sum(v) for v in pixel_counts.values()]
    assert exclude_classes is None or type(exclude_classes) == list

    # Remove specified classes
    if exclude_classes:
        keep_indices = np.where(
            [c not in set(exclude_classes) for c in class_names]
        )[0]
        class_names = [class_names[i] for i in keep_indices]
        values_list = [values_list[i] for i in keep_indices]
        seg_counts = [seg_counts[i] for i in keep_indices]
        pixel_counts = [pixel_counts[i] for i in keep_indices]

    # Left plot
    plt.subplots(1, 3, figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.barh(range(len(class_names)), pixel_counts)
    plt.gca().set_yticks(range(len(class_names)))
    plt.gca().set_yticklabels(class_names)
    plt.xlabel("Number of pixels per class")
    plt.title("Distribution of pixel labels")

    # Middle plot
    plt.subplot(1, 3, 2)
    plt.barh(range(len(class_names)), seg_counts)
    plt.gca().set_yticks(range(len(class_names)))
    plt.gca().set_yticklabels(class_names)
    plt.xlabel("Number of segments per class")
    plt.title("Distribution of segment labels")

    # Right plot
    plt.subplot(1, 3, 3)
    plt.hist(
        values_list, nr_bins, label=class_names, histtype="barstacked",
    )
    plt.title("Distribution of segment sizes (stacked bar chart)")
    plt.legend()
    plt.ylabel("Number of segments")
    plt.xlabel("Segment sizes [area in pixel]")

    if show:
        plt.show()


def plot_confusion_matrix(
    cmat: np.ndarray,
    cmat_norm: np.ndarray,
    classes: List[str],
    show: bool = True,
    figsize: Tuple[int, int] = (16, 4),
) -> None:
    """ Plot the confusion matrices.

    Args:
        cmat: confusion matrix (with raw pixel counts)
        cmat_norm: normalized confusion matrix
        classes: list of class names
        show: set to true to call matplotlib's show()
        figsize: figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ConfusionMatrixDisplay(cmat, classes).plot(
        ax=ax1,
        cmap=cm.get_cmap("Blues"),
        xticks_rotation="vertical",
        values_format="d",
    )
    ConfusionMatrixDisplay(cmat_norm, classes).plot(
        ax=ax2, cmap=cm.get_cmap("Blues"), xticks_rotation="vertical"
    )
    ax1.set_title("Confusion matrix")
    ax2.set_title("Normalized confusion matrix")

    if show:
        plt.show()
