# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from constants import TEMP_DIR
from utils_ic.datasets import Urls, unzip_url
from utils_ic.image_conversion import im2base64, ims2strlist


def test_ims2strlist():
    """ Tests extraction of image content and conversion into string"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    data_path = unzip_url(Urls.fridge_objects_path, TEMP_DIR, exist_ok=True)
    im_list = [
        os.path.join(data_path, "can", "1.jpg"),
        os.path.join(data_path, "carton", "62.jpg"),
    ]
    im_string_list = ims2strlist(im_list)
    assert isinstance(im_string_list, list)


def test_im2base64():
    """ Tests extraction of image content and conversion into bytes"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    data_path = unzip_url(Urls.fridge_objects_path, TEMP_DIR, exist_ok=True)
    im_name = os.path.join(data_path, "can", "1.jpg")
    im_content = im2base64(im_name)
    assert isinstance(im_content, bytes)
