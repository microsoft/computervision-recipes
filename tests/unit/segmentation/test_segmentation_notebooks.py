# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb

# Parameters
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
def test_01_notebook_run(segmentation_notebooks, tiny_seg_data_path): 
    notebook_path = segmentation_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            EPOCHS_HEAD=1, 
            EPOCHS_FULL=1,
            IM_SIZE=50,
            DATA_PATH=tiny_seg_data_path
            ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    overall_accuracy = nb_output.scraps["validation_overall_accuracy"].data
    class_accuracies = nb_output.scraps["validation_class_accuracies"].data
    assert len(class_accuracies) == 5
