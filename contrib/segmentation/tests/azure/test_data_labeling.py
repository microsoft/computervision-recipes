from typing import Dict

from src.azure.data_labeling import aml_coco_labels_to_standard_coco


def test_aml_coco_labels_to_standard_coco(
    aml_labels_json: Dict, standard_labels_json: Dict
):
    labels_json = aml_coco_labels_to_standard_coco(aml_labels_json)

    assert labels_json["annotations"] != aml_labels_json["annotations"]
    assert labels_json == standard_labels_json
