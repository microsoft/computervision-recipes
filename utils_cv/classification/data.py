# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Union
from urllib.parse import urljoin

from fastai.vision import ItemList
from PIL import Image
from tqdm import tqdm


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/"

    # ImageNet labels Keras is using
    imagenet_labels_json = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

    # traditional datasets
    fridge_objects_path = urljoin(base, "fridgeObjects.zip")
    fridge_objects_watermark_path = urljoin(base, "fridgeObjectsWatermark.zip")
    fridge_objects_tiny_path = urljoin(base, "fridgeObjectsTiny.zip")
    fridge_objects_watermark_tiny_path = urljoin(
        base, "fridgeObjectsWatermarkTiny.zip"
    )
    fridge_objects_negatives_path = urljoin(base, "fridgeObjectsNegative.zip")
    fridge_objects_negatives_tiny_path = urljoin(
        base, "fridgeObjectsNegativeTiny.zip"
    )

    # multilabel datasets
    multilabel_fridge_objects_path = urljoin(
        base, "multilabelFridgeObjects.zip"
    )
    multilabel_fridge_objects_watermark_path = urljoin(
        base, "multilabelFridgeObjectsWatermark.zip"
    )
    multilabel_fridge_objects_tiny_path = urljoin(
        base, "multilabelFridgeObjectsTiny.zip"
    )
    multilabel_fridge_objects_watermark_tiny_path = urljoin(
        base, "multilabelFridgeObjectsWatermarkTiny.zip"
    )

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]


def imagenet_labels() -> list:
    """List of ImageNet labels with the original index.

    Returns:
         list: ImageNet labels
    """
    labels = requests.get(Urls.imagenet_labels_json).json()
    return [labels[str(k)][1] for k in range(len(labels))]


def downsize_imagelist(
    im_list: ItemList, out_dir: Union[Path, str], dim: int = 500
):
    """Aspect-ratio preserving down-sizing of each image in the ImageList {im_list}
    so that min(width,height) is at most {dim} pixels.
    Writes each image to the directory {out_dir} while preserving the original
    subdirectory structure.

    Args:
        im_list: Fastai ItemList object containing image paths.
        out_dir: Output root location.
        dim: maximum image dimension (width/height) after resize
    """
    assert (
        len(im_list.items) > 0
    ), "Input ImageList does not contain any images."

    # Find parent directory which all images have in common
    im_paths = [str(s) for s in im_list.items]
    src_root_dir = os.path.commonprefix(im_paths)

    # Loop over all images
    for src_path in tqdm(im_list.items):
        # Load and optionally down-size image
        im = Image.open(src_path).convert("RGB")
        scale = float(dim) / min(im.size)
        if scale < 1.0:
            new_size = [int(round(f * scale)) for f in im.size]
            im = im.resize(new_size, resample=Image.LANCZOS)

        # Write image
        src_rel_path = os.path.relpath(src_path, src_root_dir)
        dst_path = os.path.join(out_dir, src_rel_path)
        assert os.path.normpath(src_rel_path) != os.path.normpath(
            dst_path
        ), "Image source and destination path should not be the same: {src_rel_path}"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        im.save(dst_path)


class LabelCsvNotFound(Exception):
    """ Exception if no csv named 'label.csv' is found in the path. """

    pass


class LabelColumnNotFound(Exception):
    """ Exception if label column not found in the CSV file. """

    pass


def is_data_multilabel(path: Union[Path, str]) -> bool:
    """ Checks if dataset is a multilabel dataset.

    A dataset is considered multilabel if it meets the following conditions:
        - a csv titled 'labels.csv' is located in the path
        - the column of the labels is titled 'labels'
        - the labels are delimited by spaces or commas
        - there exists at least one image that maps to 2 or more labels

    Args:
        path: path to the dataset

    Raises:
        MultipleCsvsFound if multiple csv files are present

    Returns:
        Whether or not the dataset is multilabel.
    """
    files = Path(path).glob("*.csv")

    if len([f for f in files]) == 0:
        return False

    csv_file_path = Path(path) / "labels.csv"

    if not csv_file_path.is_file():
        raise LabelCsvNotFound

    df = pd.read_csv(csv_file_path)

    if "labels" not in df.columns:
        raise LabelColumnNotFound

    labels = df["labels"].str.split(" ", n=1, expand=True)

    if len(labels.columns) <= 1:
        return False

    return True
