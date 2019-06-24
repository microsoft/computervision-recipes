# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
from pathlib import Path
from utils_cv.common.image import (
    im_width,
    im_height,
    im_width_height,
    im2base64,
    ims2strlist,
)


def test_im_width(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path) / "can" / "1.jpg"
    assert (
        im_width(im_path) == 499
    ), "Expected image width of 499, but got {}".format(im_width(im_path))
    im = np.zeros((100, 50))
    assert im_width(im) == 50, "Expected image width of 50, but got ".format(
        im_width(im)
    )


def test_im_height(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path) / "can" / "1.jpg"
    assert (
        im_height(im_path) == 665
    ), "Expected image height of 665, but got ".format(im_width(60))
    im = np.zeros((100, 50))
    assert (
        im_height(im) == 100
    ), "Expected image height of 100, but got ".format(im_width(im))


def test_im_width_height(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path) / "can" / "1.jpg"
    w, h = im_width_height(im_path)
    assert w == 499 and h == 665
    im = np.zeros((100, 50))
    w, h = im_width_height(im)
    assert w == 50 and h == 100


def test_ims2strlist(tiny_ic_data_path):
    """ Tests extraction of image content and conversion into string"""
    im_list = [
        os.path.join(tiny_ic_data_path, "can", "1.jpg"),
        os.path.join(tiny_ic_data_path, "carton", "34.jpg"),
    ]
    im_string_list = ims2strlist(im_list)
    assert isinstance(im_string_list, list)
    assert len(im_string_list) == len(im_list)
    for im_str in im_string_list:
        assert isinstance(im_str, str)


def test_im2base64(tiny_ic_data_path):
    """ Tests extraction of image content and conversion into bytes"""
    im_name = os.path.join(tiny_ic_data_path, "can", "1.jpg")
    im_content = im2base64(im_name)
    assert isinstance(im_content, bytes)
