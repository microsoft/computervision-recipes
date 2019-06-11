# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.similarity.metrics import (
    positive_image_rank_list,
    recall_at_k,
)


def test_positive_image_rank_list(testing_im_list):
    similarity_tuple = [
        (im_path, testing_im_list.index(im_path) * 1.0)
        for im_path in testing_im_list
    ]
    similarity_tuple_list = [similarity_tuple] * 3
    assert positive_image_rank_list(similarity_tuple_list) == [1, 1, 1]


def test_recall_at_k():
    rank_list = list(range(10))
    threshold = 5
    assert recall_at_k(rank_list, threshold) == 60.0
