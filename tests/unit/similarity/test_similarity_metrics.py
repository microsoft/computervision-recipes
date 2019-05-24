# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from utils_cv.similarity.metrics import (
    positive_image_median_rank,
    positive_in_top_k,
)
from utils_cv.common.data import data_path


def test_positive_image_median_rank():
    im_list = ["im1.jpg", "im2.jpg", "im3.jpg", "img4.jpg", "img5.jpg"]
    similarity_tuple = [
        (os.path.join(data_path(), im_name), im_list.index(im_name) * 1.0)
        for im_name in im_list
    ]
    similarity_tuple_list = [similarity_tuple] * 3
    assert positive_image_median_rank(similarity_tuple_list)[0] == [1, 1, 1]
    assert positive_image_median_rank(similarity_tuple_list)[1] == 1


def test_positive_in_top_k():
    rank_list = list(range(10))
    threshold = 5
    assert positive_in_top_k(rank_list, threshold) == 60.0
