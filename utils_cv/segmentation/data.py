# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import List
from urllib.parse import urljoin


class Urls:
    # base url
    base = "https://cvbp.blob.core.windows.net/public/datasets/image_segmentation/"

    # traditional datasets
    fridge_objects_path = urljoin(base, "segFridgeObjects.zip")
    fridge_objects_tiny_path = urljoin(base, "segFridgeObjectsTiny.zip")

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]
