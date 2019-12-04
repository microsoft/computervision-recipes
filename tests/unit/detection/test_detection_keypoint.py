# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from utils_cv.detection.keypoint import COCOKeypoints


@pytest.fixture(scope="session")
def od_sample_keypoint_with_meta():
    keypoints = np.array([
        [
            [10.0, 20.0, 2],
            [20.0, 20.0, 2],
        ],
        [
            [20.0, 10.0, 2],
            [0, 0, 0],
        ],
        [
            [30.0, 30.0, 2],
            [40.0, 40.0, 2],
        ],
        [
            [40.0, 10.0, 2],
            [50.0, 50.0, 2],
        ],
    ])
    keypoint_meta = {
        "category": "dummy",
        "labels": ["left", "right"],
        "skeleton": [[0, 1], ],
    }
    lines = [
        [10.0, 20.0, 20.0, 20.0],
        [30.0, 30.0, 40.0, 40.0],
        [40.0, 10.0, 50.0, 50.0],
    ]

    return keypoints, keypoint_meta, lines


def test_cocokeypoints(od_sample_keypoint_with_meta):
    keypoints, keypoint_meta, lines = od_sample_keypoint_with_meta

    # test init
    k = COCOKeypoints(keypoints, keypoint_meta)
    assert np.all(k.keypoints == keypoints)
    assert k.meta == keypoint_meta

    # test get_lints
    assert k.get_lines() == lines
