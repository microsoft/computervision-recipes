# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for visualizations
"""
import matplotlib.pyplot as plt
import cv2
from typing import List, Union, Tuple
from pathlib import Path


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
