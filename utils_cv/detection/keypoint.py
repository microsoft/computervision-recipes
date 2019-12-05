# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List
import numpy as np


class Keypoints:
    """ Keypoints class for keypoint detection. """

    def __init__(
        self,
        keypoints: np.ndarray,
        meta: Dict,
    ):
        """
        Args:
            keypoints: keypoints array of shape (N, num_keypoints, 3), where N
                is the number of objects.  3 means x, y and visibility.  0 for
                visibility means invisible.
            meta: a dict includes "point_num" and "skeleton".  "point_num"
                is an integer indicating the number of predefined keypoints.
                "skeleton" is a list of connections between each predefined
                keypoints.
        """
        self.meta = meta
        self.keypoints = keypoints

        # convert self.keypoints into correct type
        self.keypoints = np.asarray(self.keypoints, dtype=np.float)
        if self.keypoints.ndim != 3:
            # shape must be (N, len(self.meta["labels"]), 3)
            self.keypoints = self.keypoints.reshape(
                (-1, self.meta["point_num"], 3)
            )

        # skeleton indexes should not be out of the range of labels
        assert np.max(np.array(self.meta["skeleton"])) < self.meta["point_num"]

        # make sure invisible points' x, y = 0
        self.keypoints[self.keypoints[..., 2] == 0] = 0

    def get_lines(self) -> List[List[float]]:
        """ Return connected lines represented by list of [x1, y1, x2, y2]. """
        joints = self.keypoints[:, self.meta["skeleton"]]
        visibles = (joints[..., 2] != 0).all(axis=2)
        bones = joints[visibles][..., :2]
        lines = bones.reshape((-1, 4)).tolist()
        return lines
