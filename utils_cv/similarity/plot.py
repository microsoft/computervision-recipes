# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import os

from PIL import Image, ImageOps


def plot_similars(similars: list, num_rows: int, num_cols: int):
    """

    Args:
        similars: (list) List of tuples (image path, distance to the query_image)
        num_rows: (int) number of rows on which to display the images
        num_cols: (int) number of columns on which to display the images

    Returns: Nothing, but displays the images which paths are provided as input

    """
    for num, (image, distance) in enumerate(similars):
        plt.subplot(num_rows, num_cols, num + 1)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["axes.titlepad"] = 2
        plt.axis("off")

        title_color = "black"
        im_name = os.path.basename(image)
        title = f"{im_name}\nrank: {num} - dist: {distance:0.3f}"

        img = Image.open(image)
        if num == 0:
            img = ImageOps.expand(img, border=15, fill="orange")
            title_color = "orange"
            title = f"Reference:\n{im_name}"

        plt.title(title, fontsize=5, color=title_color)
        plt.imshow(img)
