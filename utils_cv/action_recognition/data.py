# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Union, List
from urllib.request import urlretrieve

from ..common.data import data_path


class _DatasetSpec:
    """ Properties of a Video Dataset. """

    def __init__(
        self,
        label_url: str,
        num_classes: int,
        data_path: Union[Path, str] = data_path(),
    ) -> None:
        self.label_url = label_url
        self.num_classes = num_classes
        self.data_path = data_path
        self._class_names = None

    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            label_filepath = os.path.join(self.data_path, "label_map.txt")
            if not os.path.isfile(label_filepath):
                os.makedirs(self.data_path, exist_ok=True)
            else:
                os.remove(label_filepath)
            urlretrieve(self.label_url, label_filepath)
            with open(label_filepath) as f:
                self._class_names = [l.strip() for l in f]
            assert len(self._class_names) == self.num_classes

        return self._class_names


class Urls:
    kinetics_label_map = "https://github.com/microsoft/ComputerVision/files/3746975/kinetics400_lable_map.txt"
    hmdb51_label_map = "https://github.com/microsoft/ComputerVision/files/3746963/hmdb51_label_map.txt"


KINETICS = _DatasetSpec(
    Urls.kinetics_label_map, 400, os.path.join("data", "kinetics400"),
)

HMDB51 = _DatasetSpec(
    Urls.hmdb51_label_map, 51, os.path.join("data", "hmdb51"),
)
