import tempfile
from os.path import join

import numpy as np
import pytest

from src.preprocessing.segmentation import (
    convert_segmentation,
    mask_generator,
    mask_reader,
    mask_writer,
)


@pytest.mark.parametrize(
    "segmentation, source_format, target_format, width, height, expected_segmentation",
    [
        (
            [
                0.6568766457362771,
                0.4782633500684619,
                0.6578894065221794,
                0.4727863989046098,
                0.6619404496657889,
                0.4764376996805112,
            ],
            "aml_coco",
            "coco",
            5456,
            3632,
            [3584, 1737, 3589, 1717, 3612, 1730],
        )
    ],
)
def test_convert_segmentation(
    segmentation, source_format, target_format, width, height, expected_segmentation
):
    original_segmentation = segmentation.copy()

    converted_segmentation = convert_segmentation(
        segmentation,
        source_format=source_format,
        target_format=target_format,
        image_height=height,
        image_width=width,
    )

    # Check that immutability was respected
    assert segmentation == original_segmentation
    assert converted_segmentation == expected_segmentation


def test_mask_generator(standard_labels_filepath: str, mocker):
    mask_gen = mask_generator(annotations_filepath=standard_labels_filepath)
    image_json, mask = next(mask_gen)
    zeroes = np.zeros(mask.shape)

    assert "id" in image_json
    assert "width" in image_json
    assert "height" in image_json

    assert not np.array_equal(zeroes, mask)


def test_mask_reader_and_writer(standard_labels_filepath: str):
    with tempfile.TemporaryDirectory() as tempdir:
        mask_filepath = join(tempdir, "mask.json")
        _, original_mask = next(mask_generator(standard_labels_filepath))

        mask_writer(original_mask, mask_filepath)
        mask = mask_reader(mask_filepath)

        assert np.array_equal(mask, original_mask)
        assert mask.sum() > 0
