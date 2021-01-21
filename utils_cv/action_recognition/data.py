# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Union, List
from urllib.request import urlretrieve
from urllib.parse import urljoin

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
    # base url
    base = "https://cvbp-secondary.z19.web.core.windows.net/datasets/action_recognition/"

    # label maps
    kinetics_label_map = "https://github.com/microsoft/ComputerVision/files/3746975/kinetics400_lable_map.txt"
    hmdb51_label_map = "https://github.com/microsoft/ComputerVision/files/3746963/hmdb51_label_map.txt"

    # milk bottle action split test files
    hmdb_train_split_1 = urljoin(base, "hmdb51_vid_train_split_1.txt")
    hmdb_test_split_1 = urljoin(base, "hmdb51_vid_test_split_1.txt")

    # testing datasets
    milk_bottle_action_path = urljoin(base, "milkBottleActions.zip")
    milk_bottle_action_minified_path = urljoin(
        base, "milkBottleActions_minified.zip"
    )

    # milk bottle action split test files
    milk_bottle_action_train_split = urljoin(
        base, "milk_bottle_actions_train_split.txt"
    )
    milk_bottle_action_test_split = urljoin(
        base, "milk_bottle_actions_test_split.txt"
    )

    # test vid
    drinking_path = urljoin(base, "drinking.mp4")

    # webcam sample vids
    webcam_vid = urljoin(base, "action_sample.mp4")
    webcam_vid_low_res = urljoin(base, "action_sample_lowRes.mp4")


KINETICS = _DatasetSpec(
    Urls.kinetics_label_map, 400, os.path.join("data", "kinetics400")
)

HMDB51 = _DatasetSpec(
    Urls.hmdb51_label_map, 51, os.path.join("data", "hmdb51")
)
