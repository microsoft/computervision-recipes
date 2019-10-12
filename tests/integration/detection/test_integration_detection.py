# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb

# Parameters
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_01_notebook_run(detection_notebooks):
    epochs = 3
    notebook_path = detection_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_losses"].data) == epochs
    assert nb_output.scraps["training_losses"].data[-1] < 0.5
    assert nb_output.scraps["training_average_precision"].data[-1] > 0.5

@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_12_notebook_run(detection_notebooks):
    epochs = 3
    notebook_path = detection_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["valid_accs"].data[-1] > 0.5
    assert len(nb_output.scraps["valid_accs"].data) == epochs
    assert len(nb_output.scraps["hard_im_scores"].data) == 10

