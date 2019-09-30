# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from utils_cv.similarity.data import comparative_set_builder


def test_comparative_set_builder(tiny_ic_databunch):
    comparative_sets = comparative_set_builder(
        tiny_ic_databunch.valid_ds, num_sets=20, num_negatives=50
    )
    assert isinstance(comparative_sets, list)
    assert len(comparative_sets) == 20
    for cs in comparative_sets:
        assert len(cs.neg_im_paths) == 50
        neg_and_pos_label_identical = np.where(
            np.array(cs.neg_labels) == cs.pos_label
        )[0]
        assert (
            len(neg_and_pos_label_identical) == 0
        ), "Negative contains at least one image with same label as the positive"
