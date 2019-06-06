# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.similarity.data import comparative_set_builder


def test_comparative_set_builder(testing_databunch):
    resulting_set = comparative_set_builder(testing_databunch)
    first_key, first_value = next(iter(resulting_set.items()))

    assert isinstance(resulting_set, dict)
    assert len(resulting_set) == len(testing_databunch.y)
    assert isinstance(first_key, str)
    assert isinstance(first_value, list)
