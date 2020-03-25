# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from utils_cv.action_recognition.data import (
    _DatasetSpec,
    Urls,
)
from utils_cv.common.data import data_path


def test__DatasetSpec_kinetics():
    """ Tests DatasetSpec initialize with kinetics classes """
    kinetics = _DatasetSpec(Urls.kinetics_label_map, 400)
    kinetics.class_names
    assert os.path.exists(str(data_path() / "label_map.txt"))


def test__DatasetSpec_hmdb():
    """ Tests DatasetSpec initialize with hmdb51 classes """
    hmdb51 = _DatasetSpec(Urls.hmdb51_label_map, 51)
    hmdb51.class_names
    assert os.path.exists(str(data_path() / "label_map.txt"))
