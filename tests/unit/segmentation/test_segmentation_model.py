# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import functools
import numpy as np

from utils_cv.segmentation.model import (
    get_ratio_correct_metric,
    predict,
    confusion_matrix,
    print_accuracies,
)


def test_get_ratio_correct_metric(seg_classes):
    fct = get_ratio_correct_metric(seg_classes)
    assert type(fct) == functools.partial


def test_predict(seg_im_mask_paths, seg_learner):
    im_path = seg_im_mask_paths[0][0]
    mask, scores = predict(im_path, seg_learner)
    assert mask.shape[0] == 50  # scores.shape[0] == 50
    assert mask.shape[1] == 50  # scores.shape[1] == 50
    assert len(scores) == 5
    for i in range(len(scores)):
        assert mask.shape[0] == scores[i].shape[0]
        assert mask.shape[1] == scores[i].shape[1]


def test_confusion_matrix(seg_learner, tiny_seg_databunch):
    cmat, cmat_norm = confusion_matrix(
        seg_learner, tiny_seg_databunch.valid_dl
    )
    assert type(cmat) == np.ndarray
    assert type(cmat_norm) == np.ndarray
    assert cmat.max() > 1.0
    assert cmat_norm.max() <= 1.0


def test_print_accuracies(seg_confusion_matrices, seg_classes):
    cmat, cmat_norm = seg_confusion_matrices
    print_accuracies(cmat, cmat_norm, seg_classes)
