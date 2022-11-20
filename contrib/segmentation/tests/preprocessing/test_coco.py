from typing import Dict

import pandas as pd

from src.preprocessing.coco import (
    coco_json_to_pandas_dataframe,
    pandas_dataframe_to_coco_json,
)


def test_pandas_dataframe_to_coco_json(annotations_df: pd.DataFrame):
    coco_json = pandas_dataframe_to_coco_json(annotations_df)

    # Check top level keys
    assert "images" in coco_json.keys()
    assert "annotations" in coco_json.keys()
    assert "categories" in coco_json.keys()

    # Check for one image if the correct keys are there
    image_json: Dict = coco_json["images"][0]
    assert "id" in image_json.keys()
    assert "width" in image_json.keys()
    assert "height" in image_json.keys()
    assert "file_name" in image_json.keys()
    assert "coco_url" in image_json.keys()
    assert "absolute_url" in image_json.keys()
    assert "date_captured" in image_json.keys()

    # Check for one annotation if correct keys are there
    annotation_json: Dict = coco_json["annotations"][0]
    assert "segmentation" in annotation_json.keys()
    assert "id" in annotation_json.keys()
    assert "category_id" in annotation_json.keys()
    assert "image_id" in annotation_json.keys()
    assert "area" in annotation_json.keys()
    assert "bbox" in annotation_json.keys()

    # Check for one category if correct keys are there
    category_json: Dict = coco_json["categories"][0]
    assert "id" in category_json.keys()
    assert "name" in category_json.keys()


def test_coco_json_and_pandas_dataframe_conversion(annotations_df: pd.DataFrame):
    serialized_annotations_json = pandas_dataframe_to_coco_json(annotations_df)
    serialized_annotations_df = coco_json_to_pandas_dataframe(
        serialized_annotations_json
    )

    assert annotations_df.equals(serialized_annotations_df)
