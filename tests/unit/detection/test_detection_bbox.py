# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
from typing import List, Optional

from utils_cv.detection.bbox import DetectionBbox, AnnotationBbox, _Bbox


@pytest.fixture(scope="function")
def basic_bbox() -> "_Bbox":
    return _Bbox(left=0, top=10, right=100, bottom=1000)


@pytest.fixture(scope="session")
def anno_bbox() -> "AnnotationBbox":
    return AnnotationBbox(left=0, top=10, right=100, bottom=1000, label_idx=0)


@pytest.fixture(scope="session")
def det_bbox() -> "DetectionBbox":
    return DetectionBbox(
        left=0, top=10, right=100, bottom=1000, label_idx=0, score=0.5
    )


def validate_bbox(
    bbox: _Bbox,
    rect: Optional[List[int]] = None
) -> None:
    if rect is None:
        rect = [0, 10, 100, 1000]
    assert [bbox.left, bbox.top, bbox.right, bbox.bottom] == rect


def validate_anno_bbox(
    bbox: AnnotationBbox,
    label_idx: int,
    rect: Optional[List[int]] = None,
    im_path: Optional[str] = None,
    label_name: Optional[str] = None
):
    validate_bbox(bbox, rect)
    assert type(bbox) == AnnotationBbox
    assert bbox.label_idx == label_idx
    assert bbox.im_path == im_path
    assert bbox.label_name == label_name


def text__bbox_init(basic_bbox):
    assert type(basic_bbox) == _Bbox
    validate_bbox(basic_bbox)


def test__bbox_from_array():
    # test `from_array()` bbox initialization method
    bbox_from_array = _Bbox.from_array([0, 10, 100, 1000])
    validate_bbox(bbox_from_array)
    # test `from_array_xywh()` bbox initialization method
    bbox_from_array_xywh = _Bbox.from_array_xywh([0, 10, 101, 991])
    validate_bbox(bbox_from_array_xywh)


def test__bbox_from_mask(od_mask_rects):
    binary_masks, mask, rects, _ = od_mask_rects
    # test `_Bbox.get_rect_from_binary_mask`
    assert _Bbox.get_rect_from_binary_mask(binary_masks[0]) == rects[0]
    # test `_Bbox.from_binary_mask`
    validate_bbox(_Bbox.from_binary_mask(binary_masks[0]), rects[0])
    # test `_Bbox.from_mask(mask)`
    for bbox, rect in zip(_Bbox.from_mask(mask), rects):
        validate_bbox(bbox, rect)


def test__bbox_basic_funcs(basic_bbox):
    # test rect()
    assert basic_bbox.rect() == [0, 10, 100, 1000]
    # test width()
    assert basic_bbox.width() == 100
    # test height()
    assert basic_bbox.height() == 990
    # test surface_area()
    assert basic_bbox.surface_area() == 99000


def test__bbox_overlap(basic_bbox):
    # test bbox that does not overlap
    non_overlapping_bbox = _Bbox(left=200, top=10, right=300, bottom=1000)
    overlap = basic_bbox.get_overlap_bbox(non_overlapping_bbox)
    assert overlap is None
    # test bbox that does overlap
    overlapping_bbox = _Bbox(left=0, top=500, right=100, bottom=2000)
    overlap = basic_bbox.get_overlap_bbox(overlapping_bbox)
    assert overlap == _Bbox(left=0, top=500, right=100, bottom=1000)


def test__bbox_crop(basic_bbox):
    # test valid crop sizes
    cropped_bbox = basic_bbox.crop(max_width=10, max_height=10)
    assert cropped_bbox.width() == 10
    assert cropped_bbox.height() == 10
    assert cropped_bbox.left == 0
    assert cropped_bbox.top == 10
    assert cropped_bbox.right == 10
    assert cropped_bbox.bottom == 20
    # test invalid crop sizes
    with pytest.raises(Exception):
        basic_bbox.crap(max_width=101, max_height=10)


def test__bbox_standardization():
    non_standard_bbox_0 = _Bbox(left=100, top=1000, right=0, bottom=10)
    validate_bbox(non_standard_bbox_0)


def test__bbox_is_valid(basic_bbox):
    assert basic_bbox.is_valid() is True
    assert _Bbox(left=0, top=0, right=0, bottom=0).is_valid() is False


def test__bbox_hflip_rects():
    width = 100
    # [left (x1), top (y1), right (x2), bottom (y2)]
    rect_list = [[0, 10, 20, 30], [30, 0, 50, 20]]
    rect_array = np.array(rect_list)
    flipped = [[79, 10, 99, 30], [49, 0, 69, 20]]
    assert _Bbox.hflip_rects(rect_list, width) == flipped
    assert _Bbox.hflip_rects(rect_array, width).tolist() == flipped


def test__bbox_hflip_vflip(basic_bbox):
    width = height = 2000
    assert basic_bbox.hflip(width).rect() == [1899, 10, 1999, 1000]
    assert basic_bbox.hflip(width).rect() == [0, 10, 100, 1000]
    assert basic_bbox.vflip(height).rect() == [0, 999, 100, 1989]
    assert basic_bbox.vflip(height).rect() == [0, 10, 100, 1000]


def test_annotation_bbox_init(anno_bbox):
    validate_anno_bbox(anno_bbox, label_idx=0)


def test_annotation_bbox_from_array():
    bbox_from_array = AnnotationBbox.from_array(
        [0, 10, 100, 1000], label_idx=0
    )
    validate_anno_bbox(bbox_from_array, label_idx=0)


def test_annotation_bbox_from_array_xywh():
    label_idx = 0
    label_name = "a"
    im_path = "1"

    bbox_from_array = AnnotationBbox.from_array_xywh(
        [0, 10, 101, 991],
        label_idx=label_idx,
        label_name=label_name,
        im_path=im_path,
    )
    validate_anno_bbox(
        bbox_from_array,
        label_idx=label_idx,
        label_name=label_name,
        im_path=im_path
    )


def test_annotation_bbox_from_arrays():
    rects = [[0, 0, 19, 9], [30, 20, 59, 39]]
    label_idx = 0
    label_name = "a"
    im_path = ["1", "2"]
    bboxes = AnnotationBbox.from_arrays(
        rects,
        label_idx=label_idx,
        label_name=label_name,
        im_path=im_path,
    )
    for b, r, p in zip(bboxes, rects, im_path):
        validate_anno_bbox(
            b,
            rect=r,
            label_idx=label_idx,
            label_name=label_name,
            im_path=p
        )


def test_annotation_bbox_from_mask(od_mask_rects):
    binary_masks, mask, rects, _ = od_mask_rects
    label_idx = 0
    # test `AnnotationBbox.from_binary_mask`
    validate_anno_bbox(
        AnnotationBbox.from_binary_mask(binary_masks[0], label_idx=label_idx),
        label_idx=label_idx,
        rect=rects[0]
    )
    # test `AnnotationBbox.from_mask`
    for b, r in zip(
        AnnotationBbox.from_mask(mask, label_idx=label_idx),
        rects
    ):
        validate_anno_bbox(b, label_idx=label_idx, rect=r)


def test_detection_bbox_init(det_bbox):
    validate_bbox(det_bbox)
    assert type(det_bbox) == DetectionBbox


def test_detection_bbox_from_array(det_bbox):
    bbox_from_array = DetectionBbox.from_array(
        [0, 10, 100, 1000], label_idx=0, score=0
    )
    validate_bbox(det_bbox)
    assert type(bbox_from_array) == DetectionBbox
