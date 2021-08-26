"""
Configuration file for pytest.
Contains a collection of fixture functions which are run when pytest starts
"""
import json
from os.path import join
from typing import Dict

import pandas as pd
import pytest

from src.preprocessing.coco import coco_json_to_pandas_dataframe


@pytest.fixture()
def aml_labels_filepath() -> str:
    return join("data", "test_data", "labels", "aml_coco_labels.json")


@pytest.fixture()
def aml_labels_json(aml_labels_filepath: str) -> Dict:
    return json.load(open(aml_labels_filepath, "r"))


@pytest.fixture()
def standard_labels_filepath() -> str:
    return join("data", "test_data", "labels", "standard_coco_labels.json")


@pytest.fixture()
def standard_labels_json(standard_labels_filepath) -> Dict:
    return json.load(open(standard_labels_filepath, "r"))


@pytest.fixture
def annotations_df(aml_labels_filepath: str) -> pd.DataFrame:
    coco_json = json.load(open(aml_labels_filepath, "r"))
    annotations_df = coco_json_to_pandas_dataframe(coco_json)

    return annotations_df
