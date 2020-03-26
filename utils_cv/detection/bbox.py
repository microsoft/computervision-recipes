# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Union


class _Bbox:
    """ Util to represent bounding boxes

    Generally speaking, you should use either the AnnotationBbox or the
    DetectionBbox that inherit from this class.

    Source:
    https://github.com/Azure/ObjectDetectionUsingCntk/blob/master/helpers.py
    """

    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = int(round(float(left)))
        self.top = int(round(float(top)))
        self.right = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    @classmethod
    def from_array(cls, arr: List[int]) -> "_Bbox":
        """ Create a Bbox object from an array [left, top, right, bottom] """
        return _Bbox(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def from_array_xywh(cls, arr: List[int]) -> "_Bbox":
        """ Create a Bbox object from an array [left, top, width, height] """
        return _Bbox(arr[0], arr[1], arr[0] + arr[2] - 1, arr[1] + arr[3] - 1)

    def __str__(self):
        return f"""\
Bbox object: [\
left={self.left}, \
top={self.top}, \
right={self.right}, \
bottom={self.bottom}]\
"""

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def rect(self) -> List[int]:
        return [self.left, self.top, self.right, self.bottom]

    def width(self) -> int:
        width = self.right - self.left
        assert width >= 0
        return width

    def height(self) -> int:
        height = self.bottom - self.top
        assert height >= 0
        return height

    def surface_area(self) -> float:
        return self.width() * self.height()

    def get_overlap_bbox(self, bbox: "_Bbox") -> Union[None, "_Bbox"]:
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlap_left = max(left1, left2)
        overlap_top = max(top1, top2)
        overlap_right = min(right1, right2)
        overlap_bottom = min(bottom1, bottom2)
        if (overlap_left > overlap_right) or (overlap_top > overlap_bottom):
            return None
        else:
            # TODO think about whether this actually works for classes that inherit _Bbox
            return _Bbox(
                overlap_left, overlap_top, overlap_right, overlap_bottom
            )

    def standardize(
        self
    ) -> None:  # NOTE: every setter method should call standardize
        left_new = min(self.left, self.right)
        top_new = min(self.top, self.bottom)
        right_new = max(self.left, self.right)
        bottom_new = max(self.top, self.bottom)
        self.left = left_new
        self.top = top_new
        self.right = right_new
        self.bottom = bottom_new

    def crop(self, max_width: int, max_height: int) -> "_Bbox":
        if max_height > self.height():
            raise Exception("crop height cannot be bigger than bbox height.")
        if max_width > self.width():
            raise Exception("crop width cannot be bigger than bbox width.")
        self.right = self.left + max_width
        self.bottom = self.top + max_height
        self.standardize()
        return self

    def is_valid(self) -> bool:
        if self.left > self.right or self.top > self.bottom:
            return False
        if (
            min(self.rect()) < -self.MAX_VALID_DIM
            or max(self.rect()) > self.MAX_VALID_DIM
        ):
            return False
        return True


class AnnotationBbox(_Bbox):
    """ Inherits from Bbox """

    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        label_idx: int,
        im_path: str = None,
        label_name: str = None,
    ):
        """ Initialize AnnotationBbox """
        super().__init__(left, top, right, bottom)
        self.set_meta(label_idx, im_path, label_name)

    def set_meta(
        self, label_idx: int, im_path: str = None, label_name: str = None
    ):
        self.label_idx = label_idx
        self.im_path = im_path
        self.label_name = label_name

    @classmethod
    def from_array(cls, arr: List[int], **kwargs) -> "AnnotationBbox":
        """ Create a Bbox object from an array [left, top, right, bottom] """
        bbox = super().from_array(arr)
        bbox.__class__ = AnnotationBbox
        bbox.set_meta(**kwargs)
        return bbox

    def __repr__(self):
        name = (
            "None"
            if self.label_name == str(self.label_idx)
            else self.label_name
        )
        return f"{{{str(self)} | <{name}> | label:{self.label_idx} | path:{self.im_path}}}"


class DetectionBbox(AnnotationBbox):
    """ Inherits from AnnotationBbox """

    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        label_idx: int,
        score: float,
        im_path: str = None,
        label_name: str = None,
    ):
        """ Initialize DetectionBbox """
        super().__init__(
            left,
            top,
            right,
            bottom,
            label_idx,
            im_path=im_path,
            label_name=label_name,
        )
        self.score = score

    @classmethod
    def from_array(cls, arr: List[int], **kwargs) -> "DetectionBbox":
        """ Create a Bbox object from an array [left, top, right, bottom]
        This function must take in a score.
        """
        score = kwargs["score"]
        del kwargs["score"]
        bbox = super().from_array(arr, **kwargs)
        bbox.__class__ = DetectionBbox
        bbox.score = score
        return bbox

    def __repr__(self):
        return f"{super().__repr__()} | score: {self.score}"


def bboxes_iou(bbox1: DetectionBbox, bbox2: DetectionBbox):
    """Compute intersection-over-union between two bounding boxes

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Returns intersection-over-union of the two bounding boxes
    """
    overlap_box = bbox1.get_overlap_bbox(bbox2)
    if overlap_box is not None:
        bbox1_area = bbox1.surface_area()
        bbox2_area = bbox2.surface_area()
        overlap_box_area = overlap_box.surface_area()
        iou = overlap_box_area / float(
            bbox1_area + bbox2_area - overlap_box_area
        )
    else:
        iou = 0
    assert iou >= 0
    return iou
