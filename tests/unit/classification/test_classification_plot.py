# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
from utils_cv.classification.plot import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_pr_roc_curves,
    plot_thresholds,
)
from utils_cv.classification.model import hamming_accuracy, zero_one_accuracy


def test_plot_threshold(multilabel_result):
    """ Test the plot_loss_threshold function """
    y_pred, y_true = multilabel_result
    plot_thresholds(hamming_accuracy, y_pred, y_true)
    plot_thresholds(zero_one_accuracy, y_pred, y_true)


@pytest.fixture(scope="module")
def binaryclass_result_1():
    # Binary-class classification testcase 1
    BINARY_Y_TRUE = [0, 0, 1, 1]
    BINARY_Y_SCORE = [0.1, 0.4, 0.35, 0.8]
    BINARY_CLASSES = [0, 1]
    return np.array(BINARY_Y_TRUE), np.array(BINARY_Y_SCORE), BINARY_CLASSES


@pytest.fixture(scope="module")
def binaryclass_result_2():
    # Binary-class classification testcase 2
    BINARY_Y_TRUE = [0, 0, 1, 1]
    BINARY_Y_SCORE = [[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]]
    BINARY_CLASSES = [0, 1]
    return np.array(BINARY_Y_TRUE), np.array(BINARY_Y_SCORE), BINARY_CLASSES


@pytest.fixture(scope="module")
def multiclass_result():
    # Multi-class classification testcase
    MULTI_Y_TRUE = [0, 0, 1, 1, 2, 2]
    MULTI_Y_SCORE = [
        [0.1, 0.9, 0.0],
        [0.4, 0.2, 0.4],
        [0.35, 0.15, 0.5],
        [0.1, 0.8, 0.1],
        [0.2, 0.5, 0.3],
        [0.0, 0.1, 0.9],
    ]
    MULTI_CLASSES = [0, 1, 2]
    return np.array(MULTI_Y_TRUE), np.array(MULTI_Y_SCORE), MULTI_CLASSES


def test_plot_roc_curve(
    binaryclass_result_1, binaryclass_result_2, multiclass_result
):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result_1
    plot_roc_curve(y_true, y_score, classes, False)
    y_true, y_score, classes = binaryclass_result_2
    plot_roc_curve(y_true, y_score, classes, False)
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_roc_curve(y_true, y_score, classes, False)


def test_plot_precision_recall_curve(
    binaryclass_result_1, binaryclass_result_2, multiclass_result
):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result_1
    plot_precision_recall_curve(y_true, y_score, classes, False)
    y_true, y_score, classes = binaryclass_result_2
    plot_precision_recall_curve(y_true, y_score, classes, False)
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_precision_recall_curve(y_true, y_score, classes, False)


def test_plot_pr_roc_curves(
    binaryclass_result_1, binaryclass_result_2, multiclass_result
):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result_1
    plot_pr_roc_curves(y_true, y_score, classes, False, (1, 1))
    y_true, y_score, classes = binaryclass_result_2
    plot_pr_roc_curves(y_true, y_score, classes, False, (1, 1))
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_pr_roc_curves(y_true, y_score, classes, False, (1, 1))
