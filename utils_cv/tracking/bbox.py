# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.detection.bbox import _Bbox


class TrackingBbox(_Bbox):
    """Inherits from _Bbox"""

    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        frame_id: int,
        track_id: int,
    ):
        """ Initialize TrackingBbox """
        super().__init__(left, top, right, bottom)
        self.frame_id = frame_id
        self.track_id = track_id

    def __repr__(self):
        return f"{{{str(self)} | frame: {self.frame_id} | track: {self.track_id}}}"
