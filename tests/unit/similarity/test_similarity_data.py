# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from utils_cv.similarity.data import comparative_set_builder


def test_comparative_set_builder(testing_im_list):
    test_im_list = [Path(im_name) for im_name in testing_im_list]

    resulting_set = comparative_set_builder(test_im_list)
    first_key, first_value = next(iter(resulting_set.items()))

    assert isinstance(resulting_set, dict)
    assert len(resulting_set) == len(testing_im_list)
    assert isinstance(first_key, str)
    assert isinstance(first_value, list)
