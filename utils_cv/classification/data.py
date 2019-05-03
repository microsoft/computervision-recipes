# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import requests
from pathlib import Path
from typing import List, Union
from urllib.parse import urljoin

from fastai.vision import ImageList
from PIL import Image
from tqdm import tqdm


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp.blob.core.windows.net/public/datasets/image_classification/"

    # ImageNet labels Keras is using
    imagenet_labels_json = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

    # datasets
    fridge_objects_path = urljoin(base, "fridgeObjects.zip")
    fridge_objects_watermark_path = urljoin(base, "fridgeObjectsWatermark.zip")
    fridge_objects_tiny_path = urljoin(base, "fridgeObjectsTiny.zip")
    fridge_objects_watermark_tiny_path = urljoin(
        base, "fridgeObjectsWatermarkTiny.zip"
    )
    multilabel_fridge_objects_path = urljoin(
        base, "multilabelFridgeObjects.zip"
    )
    unlabeled_objects_path = urljoin(base, "unlabeledNegative.zip")
    food_101_subset_path = urljoin(base, "food101Subset.zip")
    fashion_texture_path = urljoin(base, "fashionTexture.zip")
    flickr_logos_32_subset_path = urljoin(base, "flickrLogos32Subset.zip")
    lettuce_path = urljoin(base, "lettuce.zip")
    recycle_path = urljoin(base, "recycle_v3.zip")
    
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
    im_list: ImageList, out_dir: Union[Path, str], dim: int = 500
):
    """Aspect-ratio preserving down-sizing of each image in the ImageList {im_list}
    so that min(width,height) is at most {dim} pixels.
    Writes each image to the directory {out_dir} while preserving the original
    subdirectory structure.

    Args:
        im_list: Fastai ImageList object.
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
        im = Image.open(src_path).convert('RGB')
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
