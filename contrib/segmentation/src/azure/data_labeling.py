import copy
from typing import Dict, List

from ..preprocessing.bbox import convert_bbox
from ..preprocessing.segmentation import convert_segmentation


def aml_coco_labels_to_standard_coco(labels_json: Dict):
    """Serialize an AML COCO labels dictionary to a standard COCO labels dictionary

    Parameters
    ----------
    labels_json : dict
        Labels in AML COCO format

    Returns
    -------
    labels_json : dict
        Labels in standard COCO format
    """
    labels_json = copy.deepcopy(labels_json)

    for annotation_json in labels_json["annotations"]:
        # Index is image_id - 1 because the ids are 1-index based
        image_json = labels_json["images"][annotation_json["image_id"] - 1]

        # Convert segmentation

        # Segmentation is nested in another array
        segmentation: List[float] = annotation_json["segmentation"][0]
        segmentation = convert_segmentation(
            segmentation,
            source_format="aml_coco",
            target_format="coco",
            image_width=image_json["width"],
            image_height=image_json["height"],
        )
        annotation_json["segmentation"] = [segmentation]

        # Convert bounding box
        bbox: List[float] = annotation_json["bbox"]
        bbox = convert_bbox(
            bbox,
            source_format="aml_coco",
            target_format="coco",
            image_width=image_json["width"],
            image_height=image_json["height"],
        )
        annotation_json["bbox"] = bbox

    return labels_json
