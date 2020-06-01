# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import List
from urllib.parse import urljoin


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp.blob.core.windows.net/public/datasets/image_segmentation/"

    # traditional datasets
    fridge_objects_path = urljoin(base, "segFridgeObjects.zip")

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]
