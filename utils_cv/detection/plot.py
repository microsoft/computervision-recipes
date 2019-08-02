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
    im_path: Union[Path, str],
    boxes: List[List[int]],
    labels: List[int],
    categories: List[str],
    rect_th: int = 2,
    text_size: float = 1,
    text_th: int = 2,
    color: Tuple[int, int, int] = (255, 0, 0),
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """ Draw image with bounding boxes.

    Args:
        im_path: the location of image path to draw
        boxes: A list of [xmin, ymin, xmax, ymax] bounding boxes to draw
        labels: A list of labels, represented by the index, for each bounding box
        categories: A list of categories to index the labels on

    Returns nothing, but plots the image with bounding boxes and labels.
    """

    # Read image with cv2
    im = cv2.imread(str(im_path))

    # Convert to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if len(boxes) > 0:
        for box, label in zip(boxes, labels):

            # reformat boxes to be consumable by cv2
            box = [(box[0], box[1]), (box[2], box[3])]

            # Draw Rectangle with the coordinates
            cv2.rectangle(im, box[0], box[1], color=color, thickness=rect_th)

            # Write the prediction class
            cv2.putText(
                im,
                categories[label],
                box[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                color=color,
                thickness=text_th,
            )

    # display the output image
    plt.figure(figsize=figsize)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    plt.show()
