from typing import Tuple, Union

import pytest

from src.preprocessing.bbox import PascalVocBoundingBox, convert_bbox


@pytest.mark.parametrize(
    "bbox, source_format, target_format, image_width, image_height, expected_bbox",
    [
        (
            (0.1, 0.2, 0.3, 0.4),
            "aml_coco",
            "pascal_voc",
            480,
            270,
            PascalVocBoundingBox(48, 54, 192, 162, True),
        ),
    ],
)
def test_convert_bbox(
    bbox: Union[Tuple, PascalVocBoundingBox],
    source_format: str,
    target_format,
    image_width: int,
    image_height: int,
    expected_bbox: Union[Tuple, PascalVocBoundingBox],
):
    bbox = convert_bbox(
        bbox,
        source_format,
        target_format,
        image_width=image_width,
        image_height=image_height,
    )

    if isinstance(expected_bbox, PascalVocBoundingBox):
        assert pytest.approx(bbox.x1, 1e-8) == expected_bbox.x1
        assert pytest.approx(bbox.y1, 1e-8) == expected_bbox.y1
        assert pytest.approx(bbox.x2, 1e-8) == expected_bbox.x2
        assert pytest.approx(bbox.y2, 1e-8) == expected_bbox.y2
