# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import os
import papermill as pm
import pytest
import scrapbook as sb

# Unless manually modified, python3 should be
# the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
def test_00_notebook_run(classification_notebooks):
    notebook_path = classification_notebooks["00"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["predicted_label"].data == "coffee_mug"
    assert nb_output.scraps["predicted_confidence"].data > 0.5


@pytest.mark.notebooks
def test_01_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_ic_data_path,
            EPOCHS=1,
            IM_SIZE=50,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 1


@pytest.mark.notebooks
def test_02_notebook_run(classification_notebooks, multilabel_ic_data_path):
    notebook_path = classification_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=multilabel_ic_data_path,
            EPOCHS=1,
            IM_SIZE=50,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 1


@pytest.mark.notebooks
def test_03_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["03"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_ic_data_path,
            MULTILABEL=False,
            MODEL_TYPE="fast_inference",
            EPOCHS_HEAD=1,
            EPOCHS_BODY=1,
            IM_SIZE=50,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 1


@pytest.mark.notebooks
def test_10_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["10"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            IM_DIR=os.path.join(tiny_ic_data_path, "can"),
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["num_images"].data == 6


@pytest.mark.notebooks
def test_11_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA=[tiny_ic_data_path],
            REPS=1,
            LEARNING_RATES=[1e-3],
            IM_SIZES=[50],
            EPOCHS=[1],
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["nr_elements"].data == 1


@pytest.mark.notebooks
def test_12_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_ic_data_path,
            EPOCHS_HEAD=1,
            EPOCHS_BODY=1,
            IM_SIZE=50,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["train_acc"].data) == 1
