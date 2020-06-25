# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb

# Parameters
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


# @pytest.mark.notebooks
# def test_01_notebook_run(segmentation_notebooks, tiny_seg_data_path):
#     notebook_path = segmentation_notebooks["01"]
#     pm.execute_notebook(
#         notebook_path,
#         OUTPUT_NOTEBOOK,
#         parameters=dict(
#             PM_VERSION=pm.__version__,
#             EPOCHS=1,
#             IM_SIZE=50,
#             DATA_PATH=tiny_seg_data_path
#             ),
#         kernel_name=KERNEL_NAME,
#     )

#     nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
#     overall_accuracy = nb_output.scraps["validation_overall_accuracy"].data
#     class_accuracies = nb_output.scraps["validation_class_accuracies"].data
#     assert len(class_accuracies) == 5


@pytest.mark.notebooks
def test_11_notebook_run(segmentation_notebooks, tiny_seg_data_path):
    notebook_path = segmentation_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            REPS = 1,
            EPOCHS=[1],
            IM_SIZE=[50],
            LEARNING_RATES = [1e-4],
            DATA_PATH=[tiny_seg_data_path]
            ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    nr_elements = nb_output.scraps["nr_elements"].data
    ratio_correct = nb_output.scraps["ratio_correct"].data
    max_duration = nb_output.scraps["max_duration"].data
    min_duration = nb_output.scraps["min_duration"].data
    assert nr_elements == 2
