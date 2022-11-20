import json
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from src.datasets.coco_utils import (
    annotation_to_mask_array,
    coco_annotations_by_image_id,
    filter_coco_json_by_category_ids,
)

BoundingBox = Union[Tuple[int, int, int, int], List[int]]


class CocoDataset:
    def __init__(
        self,
        labels_filepath: str,
        root_dir: str,
        classes: List[int],
        annotation_format: str,
    ):
        coco_json: Dict = json.load(open(labels_filepath, "r"))
        coco_json = filter_coco_json_by_category_ids(coco_json, classes)

        self.root_dir = root_dir
        self.images: List[Dict] = coco_json["images"]
        self.annotations: List[Dict] = coco_json["annotations"]
        self.categories: List[Dict] = coco_json["categories"]
        self.classes = classes
        self.annotation_format = annotation_format
        self.annotations_by_image_id = coco_annotations_by_image_id(coco_json)
        self.images_by_image_id: Dict[int, Dict] = {
            image_json["id"]: image_json for image_json in self.images
        }

    def get_semantic_segmentation_info_for_image(
        self, image_id: int
    ) -> Tuple[np.ndarray, List[BoundingBox], np.ndarray, List[int]]:
        """Get the objects needed to perform semantic segmentation

        Parameters
        ----------
        image_id : int
            ID of image in dataset

        Returns
        -------
        image : np.ndarray
        bboxes : List[BoundingBox]
        mask : np.ndarray
        class_labels : List[int]
        """
        image_json = self.images_by_image_id[image_id]
        image_filepath = join(self.root_dir, image_json["file_name"])
        image = Image.open(image_filepath).convert("RGB")
        image = np.array(image).astype("uint8")
        height, width, _ = image.shape

        image_id: int = image_json["id"]
        annotations = self.annotations_by_image_id[image_id]
        bboxes: List[BoundingBox] = [ann["bbox"] for ann in annotations]
        class_labels: List[int] = [ann["category_id"] for ann in annotations]

        mask = annotation_to_mask_array(
            width=width,
            height=height,
            annotations=annotations,
            classes=self.classes,
            annotation_format=self.annotation_format,
        )

        return image, bboxes, mask, class_labels
