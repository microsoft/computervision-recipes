# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List
from urllib.parse import urljoin


class Urls:
    base = "https://cvbp-secondary.z19.web.core.windows.net/datasets/tracking/"

    cans_path = urljoin(base, "cans.zip")
    fridge_objects_path = urljoin(base, "odFridgeObjects_FairMOT-Format.zip")
    carcans_annotations_path = urljoin(base, "carcans_vott-csv-export.zip")

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]
