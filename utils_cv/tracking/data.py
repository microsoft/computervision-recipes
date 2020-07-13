# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List
from urllib.parse import urljoin


class Urls:
    datasets_base = "https://cvbp.blob.core.windows.net/public/datasets/tracking/"
    models_base = "https://cvbp.blob.core.windows.net/public/models/tracking/"

    cans_path = urljoin(datasets_base, "cans.zip")
    fridge_objects_path = urljoin(datasets_base, "odFridgeObjects_FairMOT-Format.zip")
    carcans_annotations_path = urljoin(datasets_base, "carcans_vott-csv-export.zip")
    mot_challenge_path = urljoin(datasets_base, "MOT17.zip")
    baseline_models_path = urljoin(models_base, "baselines.zip")

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]
