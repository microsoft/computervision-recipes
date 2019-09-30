# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import AuthenticationException
from azureml.core import Workspace


def get_auth():
    """
    Method to get the correct Azure ML Authentication type

    Always start with CLI Authentication and if it fails, fall back
    to interactive login
    """
    try:
        auth_type = AzureCliAuthentication()
        auth_type.get_authentication_header()
    except AuthenticationException:
        auth_type = InteractiveLoginAuthentication()

    return auth_type


def get_or_create_workspace(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    workspace_region: str,
) -> Workspace:
    """
    Returns workspace if one exists already with the name
    otherwise creates a new one.

    Args
    subscription_id: Azure subscription id
    resource_group: Azure resource group to create workspace and related resources
    workspace_name: name of azure ml workspac
    workspace_region: region for workspace
    """

    try:
        # get existing azure ml workspace
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=get_auth(),
        )

    except Exception:
        # this call might take a minute or two.
        print("Creating new workspace")
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            create_resource_group=True,
            location=workspace_region,
            auth=get_auth(),
        )

    return ws
