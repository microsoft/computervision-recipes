# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from utils_cv.action_recognition.data import _DatasetSpec
from utils_cv.common.data import data_path


def test__DatasetSpec_kinetics():
    """ Tests DatasetSpec initialize with kinetics classes """
    kinetics = _DatasetSpec(
        "https://github.com/microsoft/ComputerVision/files/3746975/kinetics400_lable_map.txt",
        400,
    )
    kinetics.class_names
    assert os.path.exists(str(data_path() / "label_map.txt"))


def test__DatasetSpec_hmdb():
    """ Tests DatasetSpec initialize with hmdb51 classes """
    hmdb51 = _DatasetSpec(
        "https://github.com/microsoft/ComputerVision/files/3746963/hmdb51_label_map.txt",
        51,
    )
    hmdb51.class_names
    assert os.path.exists(str(data_path() / "label_map.txt"))
