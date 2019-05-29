# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path
from PIL import Image, ImageOps


def plot_similars(
    similars: list,
    num_rows: int,
    num_cols: int,
    reference_color: str = "orange",
    im_info_font_size: int = 5,
):
    """Displays the images which paths are provided as input

    Args:
        similars: (list) List of tuples (image path, distance to the query_image)
        num_rows: (int) number of rows on which to display the images
        num_cols: (int) number of columns on which to display the images
        reference_color: (str) Color of frame and text of reference image
        im_info_font_size: (int) Size of image titles - Defaults to 5

    Returns: Nothing, but generates a plot

    """
    for num, (image, distance) in enumerate(similars):
        plt.subplot(num_rows, num_cols, num + 1)
        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["axes.titlepad"] = 1
        plt.subplots_adjust(hspace=0.5)

        plt.axis("off")

        title_color = "black"
        im_name = os.path.basename(image)
        title = f"{im_name}\nrank: {num}\ndist: {distance:0.2f}"

        img = Image.open(image)
        if num == 0:
            img = ImageOps.expand(img, border=15, fill=reference_color)
            title_color = reference_color
            title = f"Reference:\n{im_name}"

        plt.title(title, fontsize=im_info_font_size, color=title_color)
        plt.imshow(img)


def plot_rank_and_set_size(ranklist: list, sets_sizes: list, both=True):
    """Displays the distribution of rank of the positive image
    across comparative sets
    If both == True, also displays the distribution of
    number of comparative images in each set

    Args:
        ranklist: (list) List of ranks of the positive example
        across comparative sets
        sets_sizes: (list) List of size of the comparative sets
        both: (bool) True if users wants to plot both subplots

    Returns: Nothing but generates a plot

    """
    plt.figure(dpi=100)
    bins = np.arange(1, max(sets_sizes) + 2, 1) - 0.5
    plt.hist(ranklist, bins=bins, alpha=0.5, label="Positive example rank")
    plt.xticks(bins + 0.5)
    plt.ylabel("Number of comparative sets")
    plt.xlabel("Rank of positive example")
    plt.title("Distribution of positive example rank across comparative sets")

    if both:
        plt.hist(
            sets_sizes, bins=bins, alpha=0.5, label="# comparative images"
        )
        plt.xticks(bins + 0.5)
        plt.legend()
        plt.xlabel("Rank of positive example  /  Number of comparative images")
        plt.title(
            "Distribution of positive example rank \n& sets size across comparative sets"
        )


def plot_comparative_set(
    compar_set: list,
    compar_num: int,
    reference_color: str = "orange",
    pos_example_color: str = "green",
    im_info_font_size: int = 5,
):
    """For a given comparative set, displays:
    1. the reference image
    2. the associated positive example
    3. 5 negative examples

    Args:
        compar_set: (list of strings) List of image paths
        compar_num: (int) Number of comparative images to display
        reference_color: (str) Color of frame and text of reference image
        pos_example_color: (str) Color of frame and text of positive example
        im_info_font_size: (int) Size of image titles - Defaults to 5

    Returns: Nothing but generates a plot

    """
    comparative_set = compar_set[0 : compar_num + 1]
    for num, im_path in enumerate(comparative_set):
        im_class = Path(im_path).parts[-2]
        plt.subplot(1, compar_num + 1, num + 1)
        plt.rcParams["axes.titlepad"] = 3

        plt.axis("off")

        title_color = "black"
        im_name = os.path.basename(im_path)

        img = Image.open(im_path)
        if num == 0:
            img = ImageOps.expand(img, border=18, fill=reference_color)
            title_color = reference_color
            title = f"Reference:\n{im_class}: {im_name}"
        elif num == 1:
            img = ImageOps.expand(img, border=18, fill=pos_example_color)
            title_color = pos_example_color
            title = f"Positive example:\n{im_class}: {im_name}"
        else:
            title = f"Negative example:\n{im_class}: {im_name}"

        plt.title(title, fontsize=im_info_font_size, color=title_color)
        plt.imshow(img)
