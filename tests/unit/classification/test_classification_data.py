# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import requests
from PIL import Image
from fastai.vision.data import ImageList

from utils_cv.classification.data import (
    downsize_imagelist,
    imagenet_labels,
    is_data_multilabel,
    Urls,
)


def test_imagenet_labels():
    # Compare first five labels for quick check
    IMAGENET_LABELS_FIRST_FIVE = (
        "tench",
        "goldfish",
        "great_white_shark",
        "tiger_shark",
        "hammerhead",
    )

    labels = imagenet_labels()
    for i in range(5):
        assert labels[i] == IMAGENET_LABELS_FIRST_FIVE[i]

    # Check total number of labels
    assert len(labels) == 1000


def test_downsize_imagelist(tiny_ic_data_path, tmp):
    im_list = ImageList.from_folder(tiny_ic_data_path)
    max_dim = 50
    downsize_imagelist(im_list, tmp, max_dim)
    im_list2 = ImageList.from_folder(tmp)
    assert len(im_list) == len(im_list2)
    for im_path in im_list2.items:
        assert min(Image.open(im_path).size) <= max_dim


def test_is_data_multilabel(tiny_multilabel_ic_data_path, tiny_ic_data_path):
    """
    Tests that multilabel classification datasets and traditional
    classification datasets are correctly identified
    """
    assert is_data_multilabel(tiny_multilabel_ic_data_path)
    assert not is_data_multilabel(tiny_ic_data_path)


def test_urls():
    # Test if all urls are valid
    all_urls = Urls.all()
    for url in all_urls:
        with requests.get(url):
            pass
