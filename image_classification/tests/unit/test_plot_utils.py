# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import papermill as pm

# Unless manually modified, python3 should be the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


def test_plot_utils():
    """Run a test-notebook that tests plot-functions"""
    pm.execute_notebook(
        os.path.join("tests", "unit", "test_plot_utils.ipynb"),
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )
