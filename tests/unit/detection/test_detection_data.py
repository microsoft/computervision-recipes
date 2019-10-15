# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import requests

from utils_cv.detection.data import coco_labels, Urls


def test_urls():
    # Test if all urls are valid
    all_urls = Urls.all()
    for url in all_urls:
        with requests.get(url):
            pass


def test_coco_labels():
    # Compare first five labels for quick check
    COCO_LABELS_FIRST_FIVE = (
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
    )

    labels = coco_labels()
    for i in range(5):
        assert labels[i] == COCO_LABELS_FIRST_FIVE[i]

    # Check total number of labels
    assert len(labels) == 91
