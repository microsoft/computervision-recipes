# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.common.gpu import linux_with_gpu, which_processor


def test_which_processor():
    # Naive test: Just run the function to see whether it works or does not work
    which_processor()


def test_linux_with_gpu():
    # Naive test: Just run the function to see whether it works or does not work
    linux_with_gpu()
