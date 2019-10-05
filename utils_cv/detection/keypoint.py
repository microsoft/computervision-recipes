# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Union
import numpy as np
import torch


class COCOKeypoints:
    """ Keypoints class for COCO datasets.

    The interpretations of different categories of COCO Keypoints are the same.
    The only difference is the labels, skeleton (connection order) for various
    categories.
    """
    def __init__(
        self,
        keypoints: Union[List[float], List[List[float]], List[List[List[float]]], np.ndarray]
    ):
        self.keypoints = np.asarray(keypoints, dtype=np.float)
        if self.keypoints.ndim != 3:
            self.keypoints = self.keypoints.reshape((-1, len(self.labels), 3))
        assert self.is_valid()
        self.standardize()

    @property
    def skeleton(self):
        """ Define how to connect keypoints. """
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value

    @property
    def labels(self):
        """ Define the names of keypoints. """
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def hflip_inds(self):
        """ Define the order of keypoints when flipped horizontally. """
        return self._hflip_inds

    @hflip_inds.setter
    def hflip_inds(self, value):
        self._hflip_inds = value

    def __eq__(self, other):
        return (self.keypoints == other.keypoints).all()

    def is_valid(self) -> bool:
        """ Make sure shape is valid. """
        # shape must be (N, len(self.labels), 3)
        if self.keypoints.shape[1:] != (len(self.labels), 3):
            return False
        return True

    def standardize(self) -> None:
        """ Make sure invisible points' x, y = 0. """
        self.keypoints[self.keypoints[..., 2] == 0] = 0

    def as_tensor(self):
        return torch.as_tensor(self.keypoints, dtype=torch.float32)

    def hflip(self, width) -> None:
        """ Flip keypoints horizontally. """
        self.keypoints = self.keypoints[:, self.hflip_inds]
        self.keypoints[..., 0] = width - self.keypoints[..., 0]

    def get_lines(self) -> List[List[float]]:
        joints = self.keypoints[:, self.skeleton]
        visibles = (joints[..., 2] != 0).all(axis=2)
        bones = joints[visibles][..., :2]
        lines = bones.reshape((-1, 4)).tolist()
        return lines


class COCOPersonKeypoints(COCOKeypoints):
    """ Util to represent keypoints. """
    skeleton = [
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
    ]
    labels = [
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
    ]
    hflip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
