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
from tempfile import TemporaryDirectory
from utils_ic.datasets import unzip_url, Urls


def path_notebooks():
    """ Returns the path of the notebooks folder. """
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
        "10_image_annotation": os.path.join(
            folder_notebooks, "10_image_annotation.ipynb"),
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


@pytest.fixture(scope="function")
def tmp(tmp_path_factory):
    """Create a function-scoped temp directory.
    Will be cleaned up after each test function.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Pytest default fixture

    Returns:
        str: Temporary directory path
    """
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="session")
def tmp_session(tmp_path_factory):
    """ Same as 'tmp' fixture but with session level scope. """
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="session")
def tiny_ic_multidata_path(tmp_session) -> List[Path]:
    """ Returns the path to multiple dataset. """
    return [
        unzip_url(
            Urls.fridge_objects_watermark_tiny_path, tmp_session, exist_ok=True
        ),
        unzip_url(Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True),
    ]


@pytest.fixture(scope="session")
def tiny_ic_data_path(tmp_session) -> Path:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True)
