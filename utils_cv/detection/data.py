# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List
from urllib.parse import urljoin


class Urls:
    # for now hardcoding base url into Urls class
    base = (
        "https://cvbp.blob.core.windows.net/public/datasets/object_detection/"
    )

    # traditional datasets
    fridge_objects_path = urljoin(base, "odFridgeObjects.zip")
    fridge_objects_watermark_path = urljoin(
        base, "odFridgeObjectsWatermark.zip"
    )
    fridge_objects_tiny_path = urljoin(base, "odFridgeObjectsTiny.zip")
    fridge_objects_watermark_tiny_path = urljoin(
        base, "odFridgeObjectsWatermarkTiny.zip"
    )


def coco_labels() -> List[str]:
    """ List of Coco labels with the original idexing.

    Reference: https://github.com/pytorch/vision/blob/master/docs/source/models.rst

    Returns:
        Coco labels
    """

    return [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
