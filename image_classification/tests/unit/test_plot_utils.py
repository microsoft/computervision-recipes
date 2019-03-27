# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
from utils_ic.plot_utils import plot_roc_curve, plot_precision_recall_curve, plot_curves


@pytest.fixture(scope="function")
def binaryclass_result():
    # Binary-class classification testcase
    BINARY_Y_TRUE = [0, 0, 1, 1]
    BINARY_Y_SCORE = [0.1, 0.4, 0.35, 0.8]
    BINARY_CLASSES = [0, 1]
    return np.array(BINARY_Y_TRUE), np.array(BINARY_Y_SCORE), BINARY_CLASSES


@pytest.fixture(scope="function")
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


def test_plot_roc_curve(binaryclass_result, multiclass_result):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result
    plot_roc_curve(y_true, y_score, classes, False)
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_roc_curve(y_true, y_score, classes, False)


def test_plot_precision_recall_curve(binaryclass_result, multiclass_result):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result
    plot_precision_recall_curve(y_true, y_score, classes, False)
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_precision_recall_curve(y_true, y_score, classes, False)


def test_plot_curves(binaryclass_result, multiclass_result):
    # Binary-class plot
    y_true, y_score, classes = binaryclass_result
    plot_curves(y_true, y_score, classes, False, (1, 1))
    # Multi-class plot
    y_true, y_score, classes = multiclass_result
    plot_curves(y_true, y_score, classes, False, (1, 1))
