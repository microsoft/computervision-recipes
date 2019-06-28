# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import platform
import pytest
import scrapbook as sb
from torch.cuda import is_available


# Parameters
KERNEL_NAME = "cv"
OUTPUT_NOTEBOOK = "output.ipynb"


# Helper function to check if GPU machine which linux.
# Integration tests are too slow on Windows/CPU machines.
def linux_with_gpu():
    is_linux = platform.system().lower() == "linux"
    has_gpu = is_available()
    return is_linux and has_gpu


@pytest.mark.notebooks
def test_01_notebook_run(similarity_notebooks, tiny_ic_data_path):
    if linux_with_gpu():
        notebook_path = similarity_notebooks["01"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(
                PM_VERSION=pm.__version__,
                DATA_PATH=tiny_ic_data_path,
            ),
            kernel_name=KERNEL_NAME,
        )
        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)

        # Typcically rank is 1, more conservative check here.
        assert nb_output.scraps["median_rank"].data <= 2
