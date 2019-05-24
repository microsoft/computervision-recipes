# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from utils_cv.similarity.data import comparative_set_builder
from utils_cv.common.data import data_path


def test_comparative_set_builder():
    im_list = ["im1.jpg", "im2.jpg", "im3.jpg", "img4.jpg", "img5.jpg"]
    test_im_list = [
        Path(os.path.join(data_path(), im_name)) for im_name in im_list
    ]
    assert isinstance(comparative_set_builder(test_im_list), dict)
    assert len(comparative_set_builder(test_im_list)) == len(im_list)
