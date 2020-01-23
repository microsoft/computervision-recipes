# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from PIL import Image
from pathlib import Path
from typing import Tuple, Union


def binarise_mask(mask: Union[np.ndarray, str, Path]) -> np.ndarray:
    """ Split the mask into a set of binary masks.

    Assume the mask is already binary masks of [N, Height, Width], or
    grayscale mask of [Height, Width] with different values
    representing different objects, 0 as background.
    """
    # get numpy array from image file
    if isinstance(mask, (str, Path)):
        mask = np.array(Image.open(mask))

    # convert to numpy array
    mask = np.asarray(mask)

    # if it is a boolean array, consider it's already binarised
    if mask.ndim == 3:
        assert np.issubdtype(mask.dtype, np.bool), "'mask' should be binary."
        return mask

    assert mask.ndim == 2, "'mask' should have at least 2 channels."
    # remove background
    obj_values = np.unique(mask)[1:]
    # get the binary masks for each color (instance)
    binary_masks = mask == obj_values[:, None, None]
    return binary_masks


def colorise_binary_mask(
    binary_mask: np.ndarray, color: Tuple[int, int, int] = (2, 166, 101)
) -> np.ndarray:
    """ Set the color for the instance in the mask. """
    # create empty RGB channels
    h = binary_mask.shape[0]
    w = binary_mask.shape[1]
    r, g, b = np.zeros([3, h, w]).astype(np.uint8)
    # set corresponding color for each channel
    r[binary_mask], g[binary_mask], b[binary_mask] = color
    # merge RGB channels
    colored_mask = np.dstack([r, g, b])
    return colored_mask


def transparentise_mask(
    colored_mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """ Return a mask with fully transparent background and alpha-transparent
    instances.

    Assume channel is the third dimension of mask, and no alpha channel.
    """
    assert (
        colored_mask.shape[2] == 3
    ), "'colored_mask' should be of 3-channels RGB."
    # convert (0, 0, 0) to (0, 0, 0, 0) and
    # all other (x, y, z) to (x, y, z, alpha*255)
    binary_mask = (colored_mask != 0).any(axis=2)
    alpha_mask = (alpha * 255 * binary_mask).astype(np.uint8)
    return np.dstack([colored_mask, alpha_mask])


def merge_binary_masks(binary_masks: np.ndarray) -> np.ndarray:
    """ Merge binary masks into one grayscale mask.

    Assume binary_masks is of [N, Height, Width].
    """
    obj_values = np.arange(len(binary_masks)) + 1
    # label mask from 1 to number of instances
    labeled_masks = binary_masks * obj_values[:, None, None]
    return np.max(labeled_masks, axis=0).astype(np.uint8)
