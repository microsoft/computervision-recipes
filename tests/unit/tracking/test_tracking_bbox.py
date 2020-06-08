# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from typing import List, Optional

from utils_cv.tracking.bbox import TrackingBbox
from ..detection.test_detection_bbox import validate_bbox

@pytest.fixture(scope="session")
def tracking_bbox() -> "TrackingBbox":
    return TrackingBbox(left=0, top=10, right=100, bottom=1000, frame_id=0, track_id=0)

def validate_tracking_bbox(
    bbox: TrackingBbox,
    frame_id: int,
    track_id: int,
    rect: Optional[List[int]] = None,
):
    validate_bbox(bbox, rect)
    assert type(bbox) == TrackingBbox
    assert bbox.frame_id == frame_id
    assert bbox.track_id == track_id