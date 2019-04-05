# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import os
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory

from fastai.vision import ImageList, imagenet_stats

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
        "deploy_on_ACI": os.path.join(
            folder_notebooks,
            "deployment",
            "01_deployment_on_" "azure_container_instances.ipynb",
        ),
    }
    return paths


@pytest.fixture
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


@pytest.fixture
def tiny_ic_data(tmp):
    """Load tiny image-classification data for a test.
    TODO refactor to use actual `tiny` data once we have it.

    Returns:
        ImageDataBunch
    """
    path = Path(unzip_url(Urls.fridge_objects_path, tmp, exist_ok=True))
    return (
        ImageList.from_folder(path)
        .split_by_rand_pct(valid_pct=0.2, seed=10)
        .label_from_folder()
        .transform(size=299)
        .databunch(bs=16)
        .normalize(imagenet_stats)
    )
