# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import numpy as np
import scrapbook as sb

# Parameters
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


# @pytest.mark.notebooks
# @pytest.mark.linuxgpu
# def test_01_notebook_integration_run(segmentation_notebooks):
#     notebook_path = segmentation_notebooks["01"]
#     pm.execute_notebook(
#         notebook_path,
#         OUTPUT_NOTEBOOK,
#         parameters=dict(PM_VERSION=pm.__version__),
#         kernel_name=KERNEL_NAME,
#     )

#     nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
#     overall_accuracy = nb_output.scraps["validation_overall_accuracy"].data
#     class_accuracies = nb_output.scraps["validation_class_accuracies"].data
#     assert len(class_accuracies) == 5
#     assert overall_accuracy >= 90
#     for acc in class_accuracies:
#         assert acc > 80


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_11_notebook_integration_run(segmentation_notebooks):
    notebook_path = segmentation_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            REPS = 1,
            ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    nr_elements = nb_output.scraps["nr_elements"].data
    ratio_correct = nb_output.scraps["ratio_correct"].data
    max_duration = nb_output.scraps["max_duration"].data
    min_duration = nb_output.scraps["min_duration"].data
    assert nr_elements == 12
    assert min_duration <= 0.8 * max_duration
    assert np.max(ratio_correct) > 0.75
