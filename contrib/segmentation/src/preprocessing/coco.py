"""
Utilities for working with COCO json files
"""
from typing import Dict

import pandas as pd


def coco_json_to_pandas_dataframe(coco_json: Dict) -> pd.DataFrame:
    """Serialize COCO json to pandas dataframe

    Parameters
    ----------
    coco_json : Dict
        JSON in COCO format

    Returns
    -------
    annotations_df : pd.DataFrame
        DataFrame with `images`, `annotations`, and `categories` information from the COCO file
    """

    # Images section
    images_df = pd.DataFrame(coco_json["images"])
    images_df = images_df.rename(
        columns={"id": "image_id", "file_name": "filepath"}
    )

    # Categories section
    categories_df = pd.DataFrame(coco_json["categories"])
    categories_df = categories_df.rename(
        columns={"id": "category_id", "name": "category_name"}
    )

    # Annotations section
    annotations_df = pd.DataFrame(coco_json["annotations"])
    annotations_df = annotations_df.merge(images_df, on="image_id")
    annotations_df = annotations_df.merge(categories_df, on="category_id")

    return annotations_df


def pandas_dataframe_to_coco_json(annotations_df: pd.DataFrame) -> Dict:
    """Serialize and write out a pandas dataframe into COCO json format

    Parameters
    ----------
    annotations_df : pd.DataFrame
        DataFrame of annotations from a COCO json file

    Returns
    -------
    coco_json : Dict
        JSON representation of the annotations dataframe
    """

    images_df = annotations_df[
        [
            "image_id",
            "width",
            "height",
            "filepath",
            "coco_url",
            "absolute_url",
            "date_captured",
        ]
    ]
    images_df = images_df.rename(
        columns={"image_id": "id", "filepath": "file_name"}
    )
    images_df = images_df.drop_duplicates()
    images = images_df.to_dict(orient="records")

    categories_df = annotations_df[["category_id", "category_name"]]
    categories_df = categories_df.rename(
        columns={"category_id": "id", "category_name": "name"}
    )
    categories_df = categories_df.drop_duplicates()
    categories = categories_df.to_dict(orient="records")

    annotations_df = annotations_df[
        ["segmentation", "id", "category_id", "image_id", "area", "bbox"]
    ]
    annotations = annotations_df.to_dict(orient="records")

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return coco_json
