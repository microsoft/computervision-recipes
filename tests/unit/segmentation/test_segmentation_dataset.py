# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import fastai
import numpy as np

from utils_cv.segmentation.dataset import (
    load_im,
    load_mask,
    read_classes,
    mask_area_sizes,
)


def test_load_im(seg_im_mask_paths, seg_im_and_mask):
    im = load_im(seg_im_mask_paths[0][0])
    assert type(im) == fastai.vision.image.Image
    im = load_im(seg_im_and_mask[0])
    assert type(im) == fastai.vision.image.Image


def test_load_mask(seg_im_mask_paths, seg_im_and_mask):
    mask = load_mask(seg_im_mask_paths[1][0])
    assert type(mask) == fastai.vision.image.ImageSegment
    mask = load_mask(seg_im_and_mask[1])
    assert type(mask) == fastai.vision.image.ImageSegment


def test_read_classes(seg_classes_path, seg_classes):
    classes = read_classes(seg_classes_path)
    assert len(classes) == len(seg_classes)
    for i in range(len(classes)):
        assert classes[i] == seg_classes[i]


def test_mask_area_sizes(tiny_seg_databunch):
    areas, pixel_counts = mask_area_sizes(tiny_seg_databunch)
    assert len(areas) == 5
    assert len(pixel_counts) == 5
    assert np.sum([np.sum(v) for v in pixel_counts.values()]) == (22 * 499 * 666)
    assert type(areas[0]) == list
    for i in range(len(areas)):
        for area in areas[i]:
            assert area > 0
