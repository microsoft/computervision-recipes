# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don’t need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import os
import pytest


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
        "10_image_annotation": os.path.join(
            folder_notebooks, "10_image_annotation.ipynb"
        ),
    }
    return paths
