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
def test_01_notebook_run(classification_notebooks):
    notebook_path = classification_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 10
    assert nb_output.scraps["training_accuracies"].data[-1] > 0.70
    assert nb_output.scraps["validation_accuracy"].data > 0.70


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_02_notebook_run(classification_notebooks):
    notebook_path = classification_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 10
    assert nb_output.scraps["training_accuracies"].data[-1] > 0.70
    assert nb_output.scraps["acc_hl"].data > 0.70
    assert nb_output.scraps["acc_zol"].data > 0.4


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_03_notebook_run(classification_notebooks):
    notebook_path = classification_notebooks["03"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_accuracies"].data) == 12
    assert nb_output.scraps["training_accuracies"].data[-1] > 0.70
    assert nb_output.scraps["validation_accuracy"].data > 0.70


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_11_notebook_run(classification_notebooks, tiny_ic_data_path):
    notebook_path = classification_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            # Speed up testing since otherwise would take ~12 minutes on V100
            DATA=[tiny_ic_data_path],
            REPS=1,
            IM_SIZES=[60, 100],
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["nr_elements"].data == 6
    assert nb_output.scraps["max_accuray"].data > 0.5
    assert (
        nb_output.scraps["max_duration"].data
        > 1.05 * nb_output.scraps["min_duration"].data
    )


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_12_notebook_run(classification_notebooks):
    notebook_path = classification_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["train_acc"].data) == 12
    assert nb_output.scraps["train_acc"].data[-1] > 0.70
    assert nb_output.scraps["valid_acc"].data[-1] > 0.60
    assert len(nb_output.scraps["negative_sample_ids"].data) > 0
