# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import os
import papermill as pm
import pytest

from utils_cv.action_recognition.dataset import VideoRecord


def check_VideoRecord(record: VideoRecord, ar_im_path: str) -> None:
    """ Checks that property methods work. """
    assert record.path is ar_im_path
    assert record.label == 0
    assert record.label_name in ("cooking", None)


def test_VideoRecord(ar_im_path) -> None:
    """ Test the video record initialization. """
    correct_input_one = [ar_im_path, 0, "cooking"]
    check_VideoRecord(VideoRecord(correct_input_one), ar_im_path)

    correct_input_two = [ar_im_path, 0]
    check_VideoRecord(VideoRecord(correct_input_two), ar_im_path)


def test_VideoRecord_invalid(ar_im_path) -> None:
    """ Test the video record initialization failure. """
    incorrect_inputs = [
        [ar_im_path, "0", "cooking"],
        [ar_im_path, "0", "cooking", "extra"],
        [ar_im_path],
        [ar_im_path, "cooking", 0],
        ["invalid/path/vid.mp4", 0, "cooking"],
        ["ar_im_path, 0, cooking"],
        "ar_im_path, 0, cooking",
    ]
    for inp in incorrect_inputs:
        with pytest.raises(Exception):
            VideoRecord(inp)
