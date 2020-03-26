# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb

# Unless manually modified, cv should be
# the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "cv"
OUTPUT_NOTEBOOK = "output.ipynb"


# ----- Image classification ----------------------------------------------------------


@pytest.mark.azuremlnotebooks
def test_ic_20_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = classification_notebooks["20"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
        ),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.azuremlnotebooks
def test_ic_21_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = classification_notebooks["21"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
        ),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.azuremlnotebooks
def test_ic_22_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = classification_notebooks["22"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
        ),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.azuremlnotebooks
def test_ic_23_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = classification_notebooks["23"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
        ),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.azuremlnotebooks
def test_ic_24_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = classification_notebooks["24"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
            MAX_NODES=2,
            MAX_TOTAL_RUNS=1,
            IM_SIZES=[30, 40],
        ),
        kernel_name=KERNEL_NAME,
    )


# # ----- Object detection ----------------------------------------------------------


@pytest.mark.azuremlnotebooks
def test_od_11_notebook_run(
    detection_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = detection_notebooks["11"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
            MAX_NODES=3,
            IM_MAX_SIZES=[200],
            LEARNING_RATES=[1e-5, 3e-3],
            UTILS_DIR="utils_cv",
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["best_accuracy"].data > 0.70


@pytest.mark.azuremlnotebooks
def test_od_20_notebook_run(
    detection_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
):
    notebook_path = detection_notebooks["20"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
        ),
        kernel_name=KERNEL_NAME,
    )
