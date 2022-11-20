import warnings
from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass
class CocoBoundingBox:
    """BoundingBox in the standard COCO format

    Each attribute is in absolute coordinates to the size of the original image

    Parameters
    ----------
    x : float
        x-coordinate of the top left of the bounding box
    y : float
        y-coordinate of the top left of the bounding box
    width : float
        Width of the bounding box
    height : float
        Height of the bounding box
    absolute : bool
        True if in absolute coordinates which corresponds to the actual pixel amount
        False if in normalized coordinates which is normalized to the image's current size
    """

    x: float
    y: float
    width: float
    height: float
    absolute: bool


@dataclass
class PascalVocBoundingBox:
    """Bounding Box in Pascal VOC format

    Parameters
    ----------
    x1 : float
        x-coordinate of the top left of the bounding box
    y1 : float
        y-coordinate of the top left of the bounding box
    x2 : float
        x-coordinate of the bottom right of the bounding box
    y2 : float
        y-coordinate of the bottom right of the bounding box
    absolute : bool
        True if in absolute coordinates which corresponds to the actual pixel amount
        False if in normalized coordinates which is normalized to the image's current size
    """

    x1: float
    y1: float
    x2: float
    y2: float
    absolute: bool


def coco_to_pascal_voc(
    coco_bbox: Tuple[int, int, int, int]
) -> Tuple[float, float, float, float]:
    """COCO to Pascal VOC Data Format Conversion

    COCO Bounding Box: (x-top-left, y-top-left, width, height)
    Pascal VOC Bounding Box: (x-top-left, y-top-left, x-bottom-right, y-bottom-right)

    Parameters
    ----------
    bbox : tuple
        COCO Bounding Box

    Returns
    -------
    pascal_voc_bbox : tuple
        Pascal VOC Bounding Box
    """
    x_top_left, y_top_left, width, height = coco_bbox

    x_bottom_right = x_top_left + width
    y_bottom_right = y_top_left + height

    return x_top_left, y_top_left, x_bottom_right, y_bottom_right


def pascal_voc_to_coco(bbox: Union[Tuple, PascalVocBoundingBox]):
    if isinstance(bbox, tuple):
        bbox = PascalVocBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], True)

    width = bbox.x2 - bbox.x1
    height = bbox.y2 - bbox.y1

    return bbox.x1, bbox.y1, width, height


def normalized_pascal_voc_bbox_to_abs_bbox(
    bbox: Union[Tuple, PascalVocBoundingBox], image_width: int, image_height: int
) -> PascalVocBoundingBox:
    """
    Pascal VOC Bounding Box with normalized coordinates (percentages based on image size)
    to absolute coordinates

    Parameters
    ----------
    bbox : tuple or PascalVocBoundingBox
        Bounding Box in Pascal VOC format with normalized coordinates
    image_width : int
        Width of image to use for absolute coordinates
    image_height : int
        Height of image to use for absolute coordinates

    Returns
    -------
    bbox : PascalVocBoundingBox
        Bounding Box with absolute coordinates based on the image
    """
    if isinstance(bbox, tuple):
        bbox = PascalVocBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], True)

    bbox.x1 *= image_width
    bbox.y1 *= image_height
    bbox.x2 *= image_width
    bbox.y2 *= image_height

    return bbox


def abs_bbox_to_normalized_pascal_voc(
    bbox: Union[Tuple, PascalVocBoundingBox], image_width: int, image_height: int
) -> PascalVocBoundingBox:
    """
    Pascal VOC Bounding Box with normalized coordinates (percentages based on image size)
    to absolute coordinates

    Parameters
    ----------
    bbox : tuple or PascalVocBoundingBox
        Bounding Box in Pascal VOC format with normalized coordinates
    image_width : int
        Width of image to use for absolute coordinates
    image_height : int
        Height of image to use for absolute coordinates

    Returns
    -------
    bbox : PascalVocBoundingBox
        Bounding Box with absolute coordinates based on the image
    """
    if isinstance(bbox, tuple):
        bbox = PascalVocBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], True)

    bbox.x1 /= image_width
    bbox.y1 /= image_height
    bbox.x2 /= image_width
    bbox.y2 /= image_height

    return bbox


def convert_bbox(
    bbox: Union[Tuple, CocoBoundingBox, PascalVocBoundingBox],
    source_format: str,
    target_format: str,
    image_width: int = None,
    image_height: int = None,
) -> List[float]:
    """Convert a Bounding Box to another format

    Parameters
    ----------
    bbox : tuple or CocoBoundingBox or PascalVocBoundingBox
        Bounding box to convert to a different format
    source_format : {'coco', 'pascal_voc', 'aml_coco'}
        Format of `bbox`
    target_format : {'coco', 'pascal_voc', 'aml_coco'}
        Format to convert `bbox` to
    image_width : int
        Width of the image that `bbox` is for.
        Required if source_format or target_format is 'aml_coco'
    image_height : int
        Height of the image that `bbox` is for
        Required if source_format or target_format is 'aml_coco'

    Returns
    -------
    bbox : list of float
        Bounding Box in format specified by `target_format`

    Raises
    ------
    ValueError
        If source_format or target_format is not one of 'coco', 'pascal_voc', or 'aml_coco'
    NotImplementedError
    """
    bbox_formats = set(["coco", "pascal_voc", "aml_coco"])
    if source_format not in bbox_formats:
        raise ValueError(f"Invalid source_format. Expected one of: {bbox_formats}")
    if target_format not in bbox_formats:
        raise ValueError(f"Invalid target_format. Expected one of: {bbox_formats}")

    if source_format == "aml_coco" or target_format == "aml_coco":
        if image_width is None or image_height is None:
            raise ValueError(
                "If source_format or target_format is 'aml_coco' then image_width and image_height must be specified"
            )

    if source_format == target_format:
        warnings.warn(
            "Parameter source_format and target_format are the same. No conversion was necessary"
        )
        return bbox

    # The intermediate bounding box should always be converted to "pascal_voc"
    # This allows the many to many conversion between all bounding box types to be done
    # through the implementation of one conversion to and from "pascal_voc"

    if source_format == "coco":
        raise NotImplementedError
    elif source_format == "pascal_voc":
        raise NotImplementedError
    elif source_format == "aml_coco":
        bbox = coco_to_pascal_voc(bbox)
        bbox = normalized_pascal_voc_bbox_to_abs_bbox(
            bbox, image_width=image_width, image_height=image_height
        )

    if target_format == "pascal_voc":
        pass
    elif target_format == "coco":
        bbox = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
    elif target_format == "aml_coco":
        bbox = abs_bbox_to_normalized_pascal_voc(
            bbox, image_width=image_width, image_height=image_height
        )

    return bbox
