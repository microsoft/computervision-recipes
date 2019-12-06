# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from utils_cv.detection.mask import (
    binarise_mask,
    colorise_binary_mask,
    transparentise_mask,
    merge_binary_masks,
)


def test_binarise_mask(od_mask_rects):
    """ Test that `binarise_mask` works. """
    binary_masks, mask, _, _ = od_mask_rects
    assert np.all(binarise_mask(mask) == binary_masks)


def test_colorise_binary_mask(od_mask_rects):
    """ Test that `colorise_binary_mask` works. """
    (binary_mask, _), _, _, _ = od_mask_rects
    foreground = 9
    background = 0
    colored_mask = colorise_binary_mask(
        binary_mask, color=(foreground, foreground, foreground)
    )
    for ch in colored_mask.transpose((2, 0, 1)):
        assert np.all(ch[binary_mask] == foreground)
        assert np.all(ch[binary_mask != True] == background)


def test_transparentise_mask(od_mask_rects):
    """ Test that `transparentise_mask` works. """
    (binary_mask, _), _, _, _ = od_mask_rects
    foreground = 9
    background = 0
    colored_mask = colorise_binary_mask(
        binary_mask, color=(foreground, foreground, foreground)
    )
    transparent_mask = transparentise_mask(colored_mask, alpha=0.7)
    assert np.all(transparent_mask[binary_mask] != background)
    assert np.all(transparent_mask[binary_mask != True] == background)


def test_merge_binary_masks(od_mask_rects):
    """ Test that `merge_binary_masks` works. """
    binary_masks, mask, _, _ = od_mask_rects
    assert np.all(merge_binary_masks(binary_masks) == mask)
