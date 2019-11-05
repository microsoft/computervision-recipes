# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import papermill as pm
import pytest
import scrapbook as sb
import torch

# Unless manually modified, python3 should be
# the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
def test_00_notebook_run(detection_notebooks):
    notebook_path = detection_notebooks["00"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["detection_bounding_box"].data) > 0


@pytest.mark.gpu
@pytest.mark.notebooks
def test_01_notebook_run(detection_notebooks, tiny_od_data_path):
    notebook_path = detection_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_data_path,
            EPOCHS=1,
            IM_SIZE=100,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_losses"].data) > 0
    training_ap = nb_output.scraps["training_average_precision"].data
    assert len(training_ap) > 0
    for d in training_ap:
        assert isinstance(d, dict)
    assert len(set([len(d) for d in training_ap])) == 1


@pytest.mark.notebooks
def test_02_notebook_run(detection_notebooks, tiny_od_mask_data_path):
    notebook_path = detection_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_mask_data_path,
            DEVICE=torch.device('cuda'),
            THRESHOLD=0.1,
            EPOCHS=1,
        ),
        kernel_name=KERNEL_NAME,
    )
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_losses"].data) > 0
    training_ap = nb_output.scraps["training_average_precision"].data
    assert len(training_ap) > 0
    for d in training_ap:
        assert isinstance(d, dict)
    assert len(set([len(d) for d in training_ap])) == 1


@pytest.mark.gpu
@pytest.mark.notebooks
def test_12_notebook_run(detection_notebooks, tiny_od_data_path):
    notebook_path = detection_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_data_path,
            EPOCHS=1,
            IM_SIZE=100,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["valid_accs"].data) == 1
    assert len(nb_output.scraps["hard_im_scores"].data) == 10
