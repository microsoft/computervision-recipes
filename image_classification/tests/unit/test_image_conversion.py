# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from utils_ic.datasets import Urls, unzip_url
from utils_ic.image_conversion import im2base64, ims2json


def test_im2base64():
    """ Tests extraction of image content and conversion into string"""
    data_path = unzip_url(Urls.fridge_objects_path, exist_ok=True)
    im_list = [
        os.path.join("can", "im_1.jpg"),
        os.path.join("carton", "im_62.jpg"),
    ]
    input_to_service = ims2json(im_list, data_path)
    assert isinstance(input_to_service, str)
    assert input_to_service[0:11] == '{"data": ["'


def test_ims2json():
    """ Tests extraction of image content and conversion into bytes"""
    data_path = unzip_url(Urls.fridge_objects_path, exist_ok=True)
    im_name = os.path.join("can", "im_1.jpg")
    im_content = im2base64(im_name, data_path)
    assert isinstance(im_content, bytes)
