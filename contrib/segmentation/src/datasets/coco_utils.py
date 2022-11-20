from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from src.preprocessing.segmentation import convert_segmentation


def rescale_annotation(
    width: int, height: int, annotation: Dict, patch_rect: Tuple[int, int, int, int]
) -> Dict:
    """rescale the source image annotation wrt the new extracted patch

    Parameters
    ----------
    width : int
        Original source image width
    height : int
        Original source image height
    annotation : Dict
        A single image annotation
    patch_rect : Tuple(int, int, int, int)
        Width and Height of the extracted patch


    Returns
    -------
    new_annotation : Dict
        The rescaled annotation
    """
    new_annotation = annotation.copy()

    new_annotation["bbox"] = [
        ((annotation["bbox"][0] * width) - patch_rect[0]) / patch_rect[2],
        ((annotation["bbox"][1] * height) - patch_rect[1]) / patch_rect[3],
        (annotation["bbox"][2] * width) / patch_rect[2],
        (annotation["bbox"][3] * height) / patch_rect[3],
    ]

    if "segmentation" in annotation:
        segmentation = annotation["segmentation"][0]
        seg_x = [
            ((x * width) - patch_rect[0]) / patch_rect[2] for x in segmentation[::2]
        ]
        seg_y = [
            ((y * height) - patch_rect[1]) / patch_rect[3] for y in segmentation[1::2]
        ]

        new_segmentation = [None] * (len(seg_x) + len(seg_y))
        new_segmentation[::2] = seg_x
        new_segmentation[1::2] = seg_y

        new_annotation["segmentation"] = [new_segmentation]

    return new_annotation


def extract_windowed_patches(
    image: object,
    annotations: List[Dict],
    patch_dimension: Tuple[int, int] = (1000, 1000),
    window_overlap: float = 0.1,
) -> List[Tuple[object, List[Dict]]]:
    """For an input image with a normalized list of annotations return
    a list of all extracted patched images and rescaled annotations

    Parameters
    ----------
    image : object
        Original source image
    annotations : List[Dict]
        List of all original source image annotations
    patch_dimension : Tuple(Int, Int)
        Width and Height of the extracted patch
    window_overlap : Float
        increment window by % of patch_dimension


    Returns
    -------
    patch_list : List[Tuple[object, List[Dict]]]
        List of all extracted patches with rescaled/centered
        image and annotations
    """
    patches = []
    width = image.width
    height = image.height

    # move window of patch_dimension on the original image
    for x in range(0, width, int(patch_dimension[0] * window_overlap)):
        for y in range(0, height, int(patch_dimension[1] * window_overlap)):
            # get patch dimension
            x = min(x, width - patch_dimension[0])
            y = min(y, height - patch_dimension[1])

            # check if patch contains at least one annotation
            patch_annotations = []
            for annotation in annotations:
                bbox = annotation["bbox"].copy()
                bbox[0] = bbox[0] * width
                bbox[1] = bbox[1] * height
                bbox[2] = bbox[2] * width
                bbox[3] = bbox[3] * height

                if (
                    bbox[0] >= x
                    and bbox[0] + bbox[2] < x + patch_dimension[0]
                    and bbox[1] >= y
                    and bbox[1] + bbox[3] < y + patch_dimension[1]
                ):

                    # rescale bbox and segments
                    rescaled_annotation = rescale_annotation(
                        width,
                        height,
                        annotation,
                        (x, y, patch_dimension[0], patch_dimension[1]),
                    )

                    patch_annotations.append(rescaled_annotation)

            if len(patch_annotations) > 0:
                # crop the image for the patch before zoom
                patch = image.crop(
                    (x, y, x + patch_dimension[0], y + patch_dimension[1])
                )

                # rescale bbox and segments

                patches.append((patch, patch_annotations))

    return patches


def annotation_to_mask_array(
    width: int,
    height: int,
    annotations: List[Dict],
    classes: List[int],
    annotation_format="aml_coco",
) -> np.ndarray:
    """Convert annotations to a mask numpy array

    Parameters
    ----------
    width : int
        Original source image width
    height : int
        Original source image height
    annotations : List[Dict]
        List of all original source image annotations
    classes : List[int]
        list of classes to use for the patch image
    annotation_format : {'coco', 'aml_coco'}
        Format that the annotations are in

    Returns
    -------
    mask_array : numpy.ndarray
        The mask array with shape `(height, width)`
    """
    mask_array = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        if int(annotation["category_id"]) in classes:
            segmentation = annotation["segmentation"][0]
            segmentation = convert_segmentation(
                segmentation,
                source_format=annotation_format,
                target_format="coco",
                image_height=height,
                image_width=width,
            )

            segmentation_array = np.array(
                [list(x) for x in zip(segmentation[::2], segmentation[1::2])],
                dtype=np.int32,
            )
            cv2.fillPoly(
                mask_array, [segmentation_array], color=(annotation["category_id"])
            )

    return mask_array


def annotation_to_mask_image(
    width: int, height: int, annotations: List[Dict], classes: List[int]
) -> object:
    """Convert annotations to a mask image

    Parameters
    ----------
    width : int
        Original source image width
    height : int
        Original source image height
    annotations : List[Dict]
        List of all original source image annotations
    classes : List[int]
        list of classes to use for the patch image

    Returns
    -------
    mask_array : object
        The mask image
    """
    mask_array = annotation_to_mask_array(width, height, annotations, classes)
    return Image.fromarray(mask_array)


def extract_windowed_patches_and_mask_images(
    image: object,
    annotations: List[Dict],
    classes: List[int],
    patch_dimension: Tuple[int, int] = (1000, 1000),
    window_overlap: float = 0.1,
) -> List[Tuple[object, object]]:
    """For an input image with a normalized list of annotations return
    a list of all extracted patched images and corresponding mask images

    Parameters
    ----------
    image : object
        Original source image
    annotations : List[Dict]
        List of all original source image annotations
    classes : List[int]
        list of classes to use for the patch image
    patch_dimension : Tuple(Int, Int)
        Width and Height of the extracted patch
    window_overlap : Float
        increment window by % of patch_dimension

    Returns
    -------
    patch_list : List[Tuple[object, object]]
        List of all extracted patches and corresponding mask images
    """

    def convert(width, height, patch):
        return (patch[0], annotation_to_mask_image(width, height, patch[1], classes))

    patch_list = extract_windowed_patches(
        image, annotations, patch_dimension, window_overlap
    )
    return [convert(patch_dimension[0], patch_dimension[1], p) for p in patch_list]


def filter_coco_annotations_by_category_ids(
    coco_json: Dict, category_ids: List[int]
) -> Dict:
    """Filter COCO annotations to only contain the given category_ids

    Parameters
    ----------
    coco_json : Dict
        COCO JSON read in as a Dictionary
    category_ids : List[int]
        Annotations containing a category_id in category_ids will be retained

    Returns
    -------
    coco_json : Dict
        COCO JSON with only the annotations for the given category_ids
    """
    coco_json = deepcopy(coco_json)
    annotations = list(
        filter(
            lambda ann: ann["category_id"] in category_ids,
            coco_json["annotations"],
        )
    )
    coco_json["annotations"] = annotations
    return coco_json


def filter_coco_json_by_category_ids(coco_json: Dict, category_ids: List[int]):
    """Filter images and annotations in COCO JSON by category_ids

    Parameters
    ----------
    coco_json : Dict
        COCO JSON read in as a Dictionary
    category_ids : List[int]
        List of category ids that the COCO JSON will retain images and annotations for

    Returns
    -------
    """
    coco_json = deepcopy(coco_json)
    coco_json["annotations"] = list(
        filter(
            lambda ann: ann["category_id"] in category_ids,
            coco_json["annotations"],
        )
    )
    annotations_by_image_id = coco_annotations_by_image_id(coco_json)
    coco_json["images"] = list(
        filter(
            lambda image: image["id"] in annotations_by_image_id,
            coco_json["images"],
        )
    )
    coco_json["categories"] = list(
        filter(lambda category: category["id"] in category_ids, coco_json["categories"])
    )

    return coco_json


def coco_annotations_by_image_id(coco_json: Dict) -> Dict[int, List]:
    """Restructure the "annotations" section of the COCO data format
    to be indexable by the image_id

    Parameters
    ----------
    coco_json : Dict
        COCO JSON read in as a Dictionary

    Returns
    -------
    annotations_by_image_id : Dict[int, List]
        Dictionary with key as the image_id to the list of annotations within the image
    """
    annotations_by_image_id: Dict[int, List] = {}
    for annotation in coco_json["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = [annotation]
        else:
            annotations_by_image_id[image_id].append(annotation)
    return annotations_by_image_id


def extract_windowed_patches_and_mask_images_sub_annotation(
    image: object,
    annotations: List[Dict],
    classes: List[int],
    patch_dimension: Tuple[int, int] = (1000, 1000),
    window_overlap: float = 0.1,
    threshold: int = 100,
) -> List[Tuple[object, object]]:
    """For an input image with a normalized list of annotations return
    a list of all extracted patched images and corresponding mask images

    Parameters
    ----------
    image : object
        Original source image
    annotations : List[Dict]
        List of all original source image annotations
    classes : List[int]
        list of classes to use for the patch image
    patch_dimension : Tuple(Int, Int)
        Width and Height of the extracted patch
    window_overlap : Float
        increment window by % of patch_dimension
    threshold : Int
        minimum number of pixels in patch mask

    Returns
    -------
    patch_list : List[Tuple[object, object]]
        List of all extracted patches and corresponding mask images
    """
    patches = []
    width = image.width
    height = image.height

    mask_array = annotation_to_mask_array(width, height, annotations, classes)
    mask_image = Image.fromarray(mask_array)
    # Get monochromatic mask array in order to count number of pixels different than background.
    # This array must also be transposed due to differences in the x,y coordinates between
    # Pillow and Numpy matrix
    mask_mono_array = np.where(mask_array > 0, 1, 0).astype("uint8").transpose()

    processed = set()

    # move window of patch_dimension on the original image
    for x in range(0, width, int(patch_dimension[0] * window_overlap)):
        for y in range(0, height, int(patch_dimension[1] * window_overlap)):
            # get patch dimension
            x = min(x, width - patch_dimension[0])
            y = min(y, height - patch_dimension[1])

            if (x, y) not in processed:
                processed.add((x, y))
                if (
                    mask_mono_array[
                        x : x + patch_dimension[0], y : y + patch_dimension[1]
                    ].sum()
                    >= threshold
                ):
                    patch_pos = (x, y, x + patch_dimension[0], y + patch_dimension[1])
                    patch_image = image.crop(patch_pos)
                    patch_mask_image = mask_image.crop(patch_pos)

                    patches.append((patch_image, patch_mask_image))

    return patches
