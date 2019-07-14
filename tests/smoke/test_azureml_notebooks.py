# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import os
import glob
import papermill as pm
import pytest
import shutil

# Unless manually modified, python3 should be
# the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "cv"
OUTPUT_NOTEBOOK = "output.ipynb"

import argparse

parser = argparse.ArgumentParser()

"""
subscription_id = "2ad17db4-e26d-4c9e-999e-adae9182530c"#"YOUR_SUBSCRIPTION_ID"
resource_group = "rijaidelnewrg1"#"YOUR_RESOURCE_GROUP_NAME"  
workspace_name = "rijaidelnewrg1ws6"#"YOUR_WORKSPACE_NAME"  
workspace_region = "eastus2" #"YOUR_WORKSPACE_REGION" #Possible values eastus, eastus2 and so on.
image_name = "rijaiamlimage3" #"YOUR_IMAGE_NAME" # without the underscores.

"""

# def create_arg_parser():
#     parser = argparse.ArgumentParser(description='Process inputs')
#     parser.add_argument("--subscription_id",
#                         help="Azure Subscription Id to create resources in")
#     parser.add_argument("--resource_group",
#                         help="Name of the resource group")
#     parser.add_argument("--workspace_name",
#                         help="Name of Azure ML Workspace")
#     parser.add_argument("--workspace_region",
#                         help="Azure region to create the workspace in")
#     parser.add_argument("--image_name",
#                         help="Name of docker image in Azure ML Workspace")
#     args = parser.parse_args()
#     return args


@pytest.mark.amlnotebooks
def test_20_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region,
    image_name,
):
    notebook_path = classification_notebooks["20_azure_workspace_setup"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
            image_name=image_name,
        ),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.azureml_test
def test_21_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region):
    notebook_path = classification_notebooks["21_deployment_on_azure_container_instances"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region),
        kernel_name=KERNEL_NAME,
    )

def test_22_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region):
    notebook_path = classification_notebooks["22_deployment_on_azure_kubernetes_service"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region),
        kernel_name=KERNEL_NAME,
    )

def test_23_notebook_run(
    classification_notebooks,
    subscription_id,
    resource_group,
    workspace_name,
    workspace_region):
    notebook_path = classification_notebooks["23_aci_aks_web_service_testing"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region),
        kernel_name=KERNEL_NAME,
    )
