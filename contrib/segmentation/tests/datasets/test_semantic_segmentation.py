from typing import List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

from src.datasets.semantic_segmentation import (
    SemanticSegmentationPyTorchDataset,
)


@pytest.mark.parametrize(
    "classes, annotation_format, patch_strategy, patch_dim, resize_dim, expected_length",
    [
        (
            [1, 2, 3, 4],
            "coco",
            "deterministic_center_crop",
            (256, 256),
            None,
            44,
        ),
        (
            [1, 2, 3, 4],
            "coco",
            "crop_all",
            (256, 256),
            None,
            3300,
        ),
        (
            [2],
            "coco",
            "deterministic_center_crop",
            (256, 256),
            None,
            5,
        ),
    ],
)
def test_semantic_segmentation_dataset(
    mocker,
    high_resolution_image: Image.Image,
    standard_labels_filepath: str,
    classes: List[int],
    annotation_format: str,
    patch_strategy: str,
    patch_dim: Optional[Tuple[int, int]],
    resize_dim: Optional[Tuple[int, int]],
    expected_length: int,
):
    mocker.patch(
        "src.datasets.semantic_segmentation.Image.open",
        return_value=high_resolution_image,
    )
    dataset = SemanticSegmentationPyTorchDataset(
        standard_labels_filepath,
        root_dir="data",
        classes=classes,
        annotation_format=annotation_format,
        patch_strategy=patch_strategy,
        patch_dim=patch_dim,
        resize_dim=resize_dim,
    )

    assert len(dataset) == expected_length

    if patch_strategy == "crop_all":
        h = high_resolution_image.height
        w = high_resolution_image.width

        # Patches are taken in a grid like fashion over the image for crop_all
        # Testing boundary cases within the grid
        boundary_indexes = [
            0,
            h - 1,
            h,
            h + 1,
            w - 1,
            w,
            w + 1,
            h * w - 1,
            h * w,
            h * w + 1,
            len(dataset) - 1,
        ]
    else:
        boundary_indexes = [0, len(dataset) - 1]

    for idx in boundary_indexes:
        image, mask = dataset[idx]
        assert image.shape == (3,) + patch_dim
        assert mask.shape == patch_dim

    # Fuzz-Testing random indexes
    random_indexes = np.random.randint(0, len(dataset), size=10)
    for idx in random_indexes:
        image, mask = dataset[idx]
        assert image.shape == (3,) + patch_dim
        assert mask.shape == patch_dim
