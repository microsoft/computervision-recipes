# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from utils_cv.similarity.data import ComparativeSet
from utils_cv.similarity.metrics import recall_at_k


def plot_distances(
    distances: list,
    num_rows: int,
    num_cols: int,
    figsize: Tuple[int, int] = None,
    im_info_font_size: int = None,
):
    """Displays the images which paths are provided as input

    Args:
        distances: List of tuples (image path, distance to the query_image)
        num_rows: Number of rows on which to display the images
        num_cols: Number of columns on which to display the images
        figsize: Figure width and height in inches
        im_info_font_size: Size of image titles

    Returns: Nothing, but generates a plot

    """
    distanceVals = [d for (im, d) in distances]
    sortOrder = np.argsort(distanceVals)

    plt.subplots(num_rows, num_cols, figsize=figsize)
    for num, index in enumerate(sortOrder[: num_rows * num_cols]):
        image, distance = distances[index]

        plt.subplot(num_rows, num_cols, num + 1)
        plt.subplots_adjust(hspace=0.2)
        plt.axis("off")

        title_color = "black"
        im_name = os.path.basename(image)
        title = f"{im_name}\nrank: {num}\ndist: {distance:0.2f}"

        img = Image.open(image)
        if num == 0 and distance < 0.01:
            title_color = "orange"
            img = ImageOps.expand(img, border=15, fill=title_color)
            title = f"Reference:\n{im_name}"

        plt.title(title, fontsize=im_info_font_size, color=title_color)
        plt.imshow(img)
        plt.figsize = (1, 1)


def plot_comparative_set(
    cs: ComparativeSet,
    num_cols: int = 5,
    figsize: Tuple[int, int] = None,
    im_info_font_size: int = None,
):
    """Displays the reference image, the associated positive example,
       and the negative examples of a specified comparative set.

    Args:
        comparative_set: Comparative set to display
        num_cols: Number of comparative images to display
        figsize: Figure width and height in inches
        im_info_font_size: Size of image titles

    Returns: Nothing but generates a plot

    """
    plt.subplots(figsize=figsize)

    # plot query image
    plt.subplot(1, num_cols, 1)
    plt.axis("off")
    im = Image.open(cs.query_im_path)
    im = ImageOps.expand(im, border=18, fill="orange")
    title = f"Query:\n{cs.pos_label}: {os.path.basename(cs.query_im_path)}"
    plt.title(title, fontsize=im_info_font_size, color="orange")
    plt.imshow(im)

    # plot query image
    plt.subplot(1, num_cols, 2)
    plt.axis("off")
    im = Image.open(cs.pos_im_path)
    im = ImageOps.expand(im, border=18, fill="green")
    title = f"Positive:\n{cs.pos_label}: {os.path.basename(cs.query_im_path)}"
    plt.title(title, fontsize=im_info_font_size, color="green")
    plt.imshow(im)

    # plot negative image
    for num, neg_im_path in enumerate(cs.neg_im_paths[: num_cols - 2]):
        plt.subplot(1, num_cols, num + 3)
        plt.axis("off")
        im = Image.open(neg_im_path)
        title = (
            f"Negative:\n{cs.pos_label}: {os.path.basename(cs.query_im_path)}"
        )
        plt.title(title, fontsize=im_info_font_size, color="black")
        plt.imshow(im)


def plot_recalls(ranks: List[int], figsize: Tuple[int, int] = None):
    """Display recall at various values of k.

    Args:
        ranks: List of ranks of the positive example across comparative sets
        figsize: Figure width and height in inches

    Returns: Nothing but generates a plot

    """
    plt.subplots(figsize=figsize)

    k_vec = range(1, max(ranks) + 1)
    recalls = [recall_at_k(ranks, k) for k in k_vec]
    plt.plot(k_vec, recalls, color="darkorange", lw=2)
    plt.xlim([0.0, max(k_vec)])
    plt.ylim([0.0, 101])
    plt.ylabel("Recall")
    plt.xlabel("Top-K")
    plt.title("Recall@k curve")


def plot_ranks_distribution(
    ranks: List[int],
    x_axis_max: int = None,
    figsize: Tuple[int, int] = (15, 6),
):
    """Displays the distribution of rank of the positive image
    across comparative sets.
    If show_set_size == True, also displays the distribution of
    number of comparative images in each set.

    Args:
        ranks: List of ranks of the positive example across comparative sets
        sets_sizes: List of size of the comparative sets
        figsize: Figure width and height in inches

    Returns: Nothing but generates a plot

    """
    plt.subplots(figsize=figsize)
    if x_axis_max is None:
        x_axis_max = max(ranks) + 1
    bins = np.arange(1, x_axis_max + 2, 1) - 0.5
    plt.hist(ranks, bins=bins, alpha=0.5, label="Positive example rank")
    plt.xticks(bins + 0.5)
    plt.ylabel("Number of comparative sets")
    plt.xlabel("Rank of positive example")
    plt.title("Distribution of positive example rank across comparative sets")
