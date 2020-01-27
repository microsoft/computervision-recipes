# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.action_recognition.data import _DatasetSpec

def test__DatasetSpec():
    kinetics = _DatasetSpec(
        "https://github.com/microsoft/ComputerVision/files/3746975/kinetics400_lable_map.txt", 400
    )


