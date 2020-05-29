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
    """ Plot an image and its ground truth mask. """
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

    print(type(im))
    print(type(mask))

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
) -> None:
    """ Plot an image, its predicted mask with associated scores, and optionally the ground truth mask. """
    im = load_im(im_or_path)
    pred_mask = pil2tensor(pred_mask, np.float32)
    max_scores = np.max(np.array(pred_scores[1:]), axis=0)
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
) -> None:
    """ Plot statistics of the ground truth masks such as number or size of segments. """
    areas = mask_area_sizes(data)
    counts = [len(i[1]) for i in areas.items()]
    class_names = [classes[i[0]] for i in areas.items()]

    # Left plot
    plt.subplots(1, 2, figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.barh(range(len(class_names)), counts)
    plt.gca().set_yticks(range(len(class_names)))
    plt.gca().set_yticklabels(class_names)
    plt.xlabel("Number of segments per class")
    plt.title("Distribution of segment labels")

    # Right plot
    plt.subplot(1, 2, 2)
    plt.hist(
        [i[1] for i in areas.items()],
        nr_bins,
        label=class_names,
        histtype="barstacked",
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
    """ Plot the confusion matrix. """
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
