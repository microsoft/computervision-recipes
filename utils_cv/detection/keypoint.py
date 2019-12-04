# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List
import numpy as np
import torch


class COCOKeypoints:
    """ Keypoints class for COCO datasets.

    The interpretations of different categories of COCO Keypoints are the
    same.  The only difference is the labels and skeleton (connection order)
    for various categories.
    """

    def __init__(
        self,
        keypoints: np.ndarray,
        meta: Dict[str, List] = None,
    ):
        """
        Args:
            keypoints:
            meta:
        """
        if meta is not None:
            self._meta = meta

        self.keypoints = keypoints
        if self.keypoints is not None:
            self._init_keypoints()

    def _init_keypoints(self):
        self.keypoints = np.asarray(self.keypoints, dtype=np.float)
        if self.keypoints.ndim != 3:
            self.keypoints = self.keypoints.reshape(
                (-1, len(self.meta["labels"]), 3)
            )
        assert self.is_valid()
        self.standardize()

    @property
    def meta(self):
        """ Define how to connect keypoints. """
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    def is_valid(self) -> bool:
        # shape must be (N, len(self.meta["labels"]), 3)
        if self.keypoints.shape[1:] != (len(self.meta["labels"]), 3):
            return False

        # indexes in skeleton should be valid
        if np.max(np.array(self.meta["skeleton"])) >= len(self.meta["labels"]):
            return False

        return True

    def standardize(self) -> None:
        """ Make sure invisible points' x, y = 0. """
        self.keypoints[self.keypoints[..., 2] == 0] = 0

    def as_tensor(self):
        return torch.as_tensor(self.keypoints, dtype=torch.float32)

    def get_lines(self) -> List[List[float]]:
        """ Return connected lines represented by list of [x1, y1, x2, y2]. """
        joints = self.keypoints[:, self.meta["skeleton"]]
        visibles = (joints[..., 2] != 0).all(axis=2)
        bones = joints[visibles][..., :2]
        lines = bones.reshape((-1, 4)).tolist()
        return lines

    @classmethod
    def to_dict(cls):
        return {
            "category": cls.meta["category"],
            "labels": cls.meta["labels"],
            "skeleton": cls.meta["skeleton"],
        }


class COCOPersonKeypoints(COCOKeypoints):
    """ Util to represent keypoints.

    The Keypoint R-CNN model in PyTorch is pretrained on
    [COCO train2017](http://cocodataset.org/#keypoints-2017).

    The format of COCO annotation file can be found at
    http://cocodataset.org/#format-data

    The keypoints details can be viewed using the following Python code:

    ```
    from pprint import pprint
    from zipfile import ZipFile
    import json
    import urllib.request

    url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'  # 242 MB
    name = url.split('/')[-1]
    urllib.request.urlretrieve(url, name)
    with ZipFile(name) as zf:
        zf.extractall()

    with open('./annotations/person_keypoints_val2017.json') as f:  # 9.6 MB
        keypoints = json.load(f)

    pprint(keypoints['categories'])
    ```
    """
    meta = {
        "category": "person",
        "skeleton": [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ],
        "labels": [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle',
        ],
    }
