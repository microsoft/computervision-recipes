# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from utils_ic.image_conversion import im2base64, ims2strlist


def test_ims2strlist(tiny_ic_data_path):
    """ Tests extraction of image content and conversion into string"""
    im_list = [
        os.path.join(tiny_ic_data_path, "can", "1.jpg"),
        os.path.join(tiny_ic_data_path, "carton", "34.jpg"),
    ]
    im_string_list = ims2strlist(im_list)
    assert isinstance(im_string_list, list)


def test_im2base64(tiny_ic_data_path):
    """ Tests extraction of image content and conversion into bytes"""
    im_name = os.path.join(tiny_ic_data_path, "can", "1.jpg")
    im_content = im2base64(im_name)
    assert isinstance(im_content, bytes)
