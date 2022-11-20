from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.datasets.coco_utils import annotation_to_mask_array


def annotation_to_mask_array_cases() -> List[Tuple]:
    width = 512
    height = 256
    category_id = 1
    top_left_mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    # Mask filling is assumed to be inclusive with the coordinates so + 1
    top_left_mask[: height // 2 + 1, : width // 2 + 1] = 1
    top_left_mask_annotation_coco_format = {
        "segmentation": [
            [
                0,
                0,
                width // 2,
                0,
                width // 2,
                height // 2,
                0,
                height // 2,
                0,
                0,
            ]
        ],
        "id": 0,
        "category_id": category_id,
        "image_id": 1,
        "area": 65536,
        "bbox": [0, 0, width // 2, height // 2],
    }

    top_left_mask_annotation_aml_coco_format = {
        "segmentation": [
            [
                0,
                0,
                0.5,
                0,
                0.5,
                0.5,
                0,
                0.5,
                0,
                0,
            ]
        ],
        "id": 0,
        "category_id": category_id,
        "image_id": 1,
        "area": 65536,
        "bbox": [0, 0, 0.5, 0.5],
    }

    top_left_mask_coco_format_case = (
        width,
        height,
        [top_left_mask_annotation_coco_format],
        [category_id],
        "coco",
        top_left_mask.copy(),
    )

    top_left_mask_aml_coco_format_case = (
        width,
        height,
        [top_left_mask_annotation_aml_coco_format],
        [category_id],
        "aml_coco",
        top_left_mask.copy(),
    )

    cases = [
        top_left_mask_coco_format_case,
        top_left_mask_aml_coco_format_case,
    ]

    return cases


@pytest.mark.parametrize(
    "width, height, annotations, classes, annotation_format, expected_mask",
    annotation_to_mask_array_cases(),
)
def test_annotation_to_mask_array(
    width: int,
    height: int,
    annotations: List[Dict],
    classes: List[int],
    annotation_format: str,
    expected_mask: np.ndarray,
):
    mask = annotation_to_mask_array(
        width, height, annotations, classes, annotation_format
    )

    assert np.array_equal(mask, expected_mask)
