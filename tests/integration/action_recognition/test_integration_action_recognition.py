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
def test_01_notebook_run(action_recognition_notebooks):
    epochs = 4
    notebook_path = action_recognition_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)

    vid_pred_accuracy = nb_output.scraps["vid_pred_accuracy"].data
    clip_pred_accuracy = nb_output.scraps["clip_pred_accuracy"].data

    assert vid_pred_accuracy > 0.3
    assert clip_pred_accuracy > 0.3
