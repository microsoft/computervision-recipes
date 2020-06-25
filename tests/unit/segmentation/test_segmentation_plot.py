# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from utils_cv.segmentation.plot import (
    plot_image_and_mask,
    plot_segmentation,
    plot_mask_stats,
    plot_confusion_matrix,
)


def test_plot_image_and_mask(seg_im_and_mask):
    plot_image_and_mask(seg_im_and_mask[0], seg_im_and_mask[1])


def test_plot_segmentation(seg_im_and_mask, seg_prediction):
    mask, scores = seg_prediction
    plot_segmentation(seg_im_and_mask[0], mask, scores)


def test_plot_mask_stats(tiny_seg_databunch, seg_classes):
    plot_mask_stats(tiny_seg_databunch, seg_classes)
    plot_mask_stats(
        tiny_seg_databunch, seg_classes, exclude_classes=["background"]
    )

   
#def test_plot_confusion_matrix(seg_confusion_matrices, seg_classes):
#    cmat, cmat_norm = seg_confusion_matrices
#    plot_confusion_matrix(cmat, cmat_norm, seg_classes)
