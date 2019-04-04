# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import os
import pytest
from pathlib import Path
from typing import List
from utils_ic.datasets import unzip_url, Urls


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "notebooks")
    )


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "00_webcam": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01_training_introduction": os.path.join(
            folder_notebooks, "01_training_introduction.ipynb"
        ),
        "02_training_accuracy_vs_speed": os.path.join(
            folder_notebooks, "02_training_accuracy_vs_speed.ipynb"
        ),
        "11_exploring_hyperparameters": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters.ipynb"
        ),
        "deploy_on_ACI": os.path.join(
            folder_notebooks,
            "deployment",
            "01_deployment_on_" "azure_container_instances.ipynb",
        ),
    }
    return paths


@pytest.fixture(scope="module")
def multidataset(tmp_path_factory) -> List[Path]:
    fp = tmp_path_factory.mktemp("multidatasets")
    return [
        unzip_url(Urls.fridge_objects_watermark_tiny_path, fp, exist_ok=True),
        unzip_url(Urls.fridge_objects_tiny_path, fp, exist_ok=True),
    ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    fp = tmp_path_factory.mktemp("dataset")
    return unzip_url(Urls.fridge_objects_tiny_path, fp, exist_ok=True)
