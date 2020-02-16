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
def test_01_notebook_run(similarity_notebooks):
    notebook_path = similarity_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["median_rank"].data <= 15


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_02_notebook_run(similarity_notebooks):
    notebook_path = similarity_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["recallAt1"].data >= 70


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_11_notebook_run(similarity_notebooks, tiny_ic_data_path):
    notebook_path = similarity_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            # Speed up testing since otherwise would take ~12 minutes on V100
            DATA_PATHS=[tiny_ic_data_path],
            REPS=1,
            IM_SIZES=[60, 70],
            LEARNING_RATES=[1e-3, 1e-4]
        ),
        kernel_name=KERNEL_NAME,
    )
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert min(nb_output.scraps["ranks"].data) <= 40


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_12_notebook_run(similarity_notebooks):
    notebook_path = similarity_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["median_rank"].data <= 5
    assert nb_output.scraps["feature_dimension"].data == 512
