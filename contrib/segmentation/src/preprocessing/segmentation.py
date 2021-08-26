import json
import warnings
from typing import List, Union

import numpy as np
import pycocotools.mask as m
from pycocotools.coco import COCO


def absolute_mask_to_normalized_mask(
    segmentation: List[float], width: int, height: int
):
    """Convert segmentation map from absolute to normalized coordinates

    Parameters
    ----------
    segmentation : list of float
        Segmentation map in absolute coordinates
    width : int
        Width of image
    height : int
        Height of image

    Returns
    -------
    segmentation : list of float
        Segmentation map converted to normalized coordinates
    """
    # This function treats the original copy of segmentation as immutable
    segmentation = segmentation.copy()

    # Segmentation is a list of even length with every 2 entries being (x, y) coordinates
    # of the next point to construct the polygon
    for i in range(0, len(segmentation), 2):
        segmentation[i] /= width
        segmentation[i + 1] /= height
    return segmentation


def normalized_mask_to_absolute_mask(
    segmentation: List[float], width: int, height: int
) -> List[float]:
    """Convert segmentation map from normalized to absolute coordinates

    Parameters
    ----------
    segmentation : list of float
        Segmentation map in normalized coordinates
    width : int
        Width of image
    height : int
        Height of image

    Returns
    -------
    segmentation : list of float
        Segmentation map converted to absolute coordinates
    """
    # This function treats the original copy of segmentation as immutable
    segmentation = segmentation.copy()

    # Segmentation is a list of even length with every 2 entries being (x, y) coordinates
    # of the next point to construct the polygon
    for i in range(0, len(segmentation), 2):
        segmentation[i] = np.round(segmentation[i] * width)
        segmentation[i + 1] = np.round(segmentation[i + 1] * height)
    return segmentation


def convert_segmentation(
    segmentation: List[Union[float, int]],
    source_format: str,
    target_format: str,
    image_width: int,
    image_height: int,
):
    """Convert a Segmentation Map to another format

    Parameters
    ----------
    segmentation : list of float or int
        Segmentation map in format `source_format`
    source_format : {'coco', 'aml_coco'}
        Format of `segmentation`
    target_format : {'coco', 'aml_coco'}
        Format of `segmentation`
    image_width : int
        Width of the image that `segmentation` is for
    image_height : int
        Height of image that `segmentation` is for

    Returns
    -------
    segmentation : list of float or int
        Segmentation map converted to `target_format`
    """
    mask_formats = set(["coco", "aml_coco", "yolo"])
    if source_format not in mask_formats:
        raise ValueError(f"Invalid source_format. Expected one of {mask_formats}")
    if target_format not in mask_formats:
        raise ValueError(f"Invalid target_format. Expected one of {mask_formats}")

    if source_format == target_format:
        warnings.warn(
            "Parameter source_format and target_format are the same. No conversion was necessary"
        )
        return segmentation

    # The intermediate segmentation mask will always be "coco"
    if source_format == "aml_coco" or source_format == "yolo":
        segmentation = normalized_mask_to_absolute_mask(
            segmentation, width=image_width, height=image_height
        )
    elif source_format == "coco":
        # Our intermediate format is coco, so we don't need to do anything
        pass

    if target_format == "aml_coco" or target_format == "yolo":
        segmentation = absolute_mask_to_normalized_mask(
            segmentation, width=image_width, height=image_height
        )
    elif target_format == "coco":
        pass

    return segmentation


def mask_generator(annotations_filepath):
    mask_builder = MaskBuilder(annotations_filepath=annotations_filepath)

    for image_id, image_json in mask_builder.coco.imgs.items():
        yield (image_json, mask_builder.construct_mask_for_image(image_id))


def mask_reader(mask_filepath: str) -> np.ndarray:
    """Read mask from filesystem.
    Masks are stored in RLE format so they are decoded before being returned

    Parameters
    ----------
    mask_filepath : str
        Filepath to read segmentation mask from

    Returns
    -------
    mask : np.ndarray
        Segmentation mask in 2D array
    """
    mask_json = json.load(open(mask_filepath, "r"))
    mask_json["counts"] = mask_json["counts"].encode()
    mask: np.ndarray = m.decode(mask_json)
    return mask


def mask_writer(mask: np.ndarray, mask_filepath: str):
    """Write segmentation masks to filesystem
    RLE is a lossless compression format that is well suited for segmentation masks which are
    by nature sparse 2D arrays

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask to write out
    mask_filepath : str
        Filepath to write segmentation mask to
    """
    mask = np.asfortranarray(mask)
    mask_json = m.encode(mask)
    mask_json["counts"] = mask_json["counts"].decode("utf-8")
    json.dump(mask_json, open(mask_filepath, "w"))


class MaskBuilder:
    def __init__(self, annotations_filepath):
        self.coco = COCO(annotation_file=annotations_filepath)
        self.category_ids = self.coco.cats.keys()

    def construct_mask_for_image(self, image_id: int) -> np.ndarray:
        """Construct segmentation mask with all annotations for image with id `image_id`

        Parameters
        ----------
        image_id : int
            Id of image to construct mask for

        Returns
        -------
        mask : np.ndarray
            Mask array with same shape as image
            Entires are the category_id for where there are instances of them
        """
        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=self.category_ids
        )
        annotations = self.coco.loadAnns(annotation_ids)
        image = self.coco.imgs[image_id]

        # Initialize a zero mask
        mask = np.zeros((int(image["height"]), int(image["width"])), dtype=np.uint8)

        # Add each annotation to the initial mask
        for i in range(len(annotations)):
            category_id = annotations[i]["category_id"]

            # The annotated mask is the same shape as our initial mask
            # Entries with the value of "category_id" indicate the presence of that
            # class in that location
            annotated_mask = self.coco.annToMask(annotations[i])
            annotated_mask = annotated_mask * category_id
            annotated_mask = annotated_mask.astype(np.uint8)

            # The masks are combined together as the 0 mask will always be overwritten by
            # actual classes
            # In the case of overlap, the class with the higher category_id is written
            mask = np.maximum(annotated_mask, mask)

        mask = mask.astype(np.uint8)
        return mask
