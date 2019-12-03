# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List
import numpy as np
import torch


class COCOKeypoints:
    """ Keypoints class for COCO datasets.

    The interpretations of different categories of COCO Keypoints are the
    same.  The only difference is the labels, skeleton (connection order) and
    hflip_inds (indexes of labels when horizontally flipped) for various
    categories.
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
        # hflip_inds should the same length as labels
        if len(self.meta["labels"]) != len(self.meta["hflip_inds"]):
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
            "hflip_inds": cls.meta["hflip_inds"],
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
        # left becomes right when flipped horizontally
        "hflip_inds": [
            0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
        ],
    }


class CartonKeypoints(COCOKeypoints):
    """ Custom keypoints of carton in the odFridgeObjects dataset. """
    meta = {
        "category": "carton",
        "skeleton": [
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [3, 7],
            [4, 6],
            [4, 8],
            [5, 6],
            [7, 8],
            [5, 7],
            [6, 8],
            [5, 9],
            [6, 10],
            [7, 11],
            [8, 12],
            [9, 10],
            [11, 12],
            [9, 11],
            [10, 12],
        ],
        "labels": [
            'lid',
            'left_top',
            'right_top',
            'left_collar',
            'right_collar',
            'left_front_shoulder',
            'right_front_shoulder',
            'left_back_shoulder',
            'right_back_shoulder',
            'left_front_bottom',
            'right_front_bottom',
            'left_back_bottom',
            'right_back_bottom',
        ],
        # left becomes right when flipped horizontally
        "hflip_inds": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11],
    }


class CanKeypoints(COCOKeypoints):
    """ Custom keypoints of can in the odFridgeObjects dataset. """
    meta = {
        "category": "can",
        "skeleton": [
            [0, 1],
            [0, 2],
            [0, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
            [4, 5],
            [6, 7],
        ],
        "labels": [
            'top_center',
            'ring_pull',
            'left_collar',
            'right_collar',
            'left_shoulder',
            'right_shoulder',
            'left_bottom',
            'right_bottom',
        ],
        # left becomes right when flipped horizontally
        "hflip_inds": [0, 1, 3, 2, 5, 4, 7, 6],
    }


class WaterBottleKeypoints(COCOKeypoints):
    """ Custom keypoints of water bottle in the odFridgeObjects dataset. """
    meta = {
        "category": "water_bottle",
        "skeleton": [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 5],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        "labels": [
            'lid_left_top',
            'lid_right_top',
            'lid_left_bottom',
            'lid_right_bottom',
            'wrapper_left_top',
            'wrapper_right_top',
            'wrapper_left_bottom',
            'wrapper_right_bottom',
        ],
        # left becomes right when flipped horizontally
        "hflip_inds": [1, 0, 3, 2, 5, 4, 7, 6],
    }


class MilkBottleKeypoints(COCOKeypoints):
    """ Custom keypoints of milk bottle in the odFridgeObjects dataset. """
    meta = {
        "category": "milk_bottle",
        "skeleton": [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 5],
            [4, 5],
        ],
        "labels": [
            'lid_left_top',
            'lid_right_top',
            'lid_left_bottom',
            'lid_right_bottom',
            'left_bottom',
            'right_bottom',
        ],
        # left becomes right when flipped horizontally
        "hflip_inds": [1, 0, 3, 2, 5, 4],
    }
