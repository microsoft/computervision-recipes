# python regular libraries
from dotenv import load_dotenv, find_dotenv
from itertools import compress
import os
from pathlib import Path
from typing import Tuple

# Azure
from azureml.core import Workspace


def update_environment_variables(parameters_list: list):
    """
    Updates the local environment variables
    This is to prevent them from being stuck with the original version
    of the .aml_env file, before it got updated
    :param parameters_list: (list of strings) List of 4 new values
    entered by the user or retrieved from the config.json file
    :return: Nothing
    """
    os.environ['SUBSCRIPTION_ID'] = parameters_list[0]
    os.environ['RESOURCE_GROUP'] = parameters_list[1]
    os.environ['WORKSPACE_REGION'] = parameters_list[2]
    os.environ['WORKSPACE_NAME'] = parameters_list[3]


def workspace_parameters_input() -> Tuple:
    """
    Turns on the interactive mode for the user to enter
    their subscription, resource group and workspace information

    :return: Tuple of information entered by the user
    """
    print("-- No *.aml_env* file could be found - Let's create one --\n"
          "-- Please provide (without quotes):")
    subscription_id = input("  > Your subscription ID: ")
    resource_group = input("  > Your resource group: ")
    workspace_name = input("  > The name of your workspace: ")
    workspace_region = input("  > The region of your workspace\n"
                             "(full list available at "
                             "https://azure.microsoft.com/en-us/"
                             "global-infrastructure/geographies/): ")

    if '' in [subscription_id, resource_group,
              workspace_name, workspace_region]:
        raise ValueError("Please provide non empty values")

    return subscription_id, resource_group, workspace_name, workspace_region


def get_or_create_dot_env_file(subscription_id: str = None,
                               resource_group: str = None,
                               workspace_name: str = None,
                               workspace_region: str = None,
                               overwrite: bool = False) -> str:
    """
    Creates a .aml_env file if it cannot find any
    in the current directory's parent hierarchy

    :param subscription_id: Subscription ID (available on Azure portal)
    :param resource_group: Resource group (created by the user)
    :param workspace_name: Workspace name (chosen by the user)
    :param workspace_region: Workspace region (e.g. "westus", "eastus2, etc.)
    :param overwrite: Whether file content should be overwritten or not
    :return: (string) Full path of the .aml_env file
    """
    # Look for already existing .aml_env file
    dotenv_path = find_dotenv('.aml_env')

    # If found and no need to overwrite it, retrieve it, else create a new one
    if dotenv_path != '' and not overwrite:
        print(f"Found the *.aml_env* file in: {dotenv_path}")
    else:
        if not overwrite:
            # if .aml_env file not found,
            # ask user to provide relevant information
            subscription_id, resource_group, workspace_name, workspace_region\
                = workspace_parameters_input()

            # if .aml_env file exists and needs to be overwritten,
            # use information from saved workspace

        home_dir = str(Path.home())
        try:
            env_file = open(os.path.join(home_dir, ".aml_env"), "w")
            sentence = "SUBSCRIPTION_ID={}\nRESOURCE_GROUP={}\n" \
                       "WORKSPACE_NAME={}\nWORKSPACE_REGION={}"\
                .format(subscription_id, resource_group,
                        workspace_name, workspace_region)
            env_file.write(sentence)
            env_file.close()

            new_environment_variables = [subscription_id, resource_group,
                                         workspace_region, workspace_name]
            update_environment_variables(new_environment_variables)
            dotenv_path = find_dotenv('.aml_env')
            print(f"The *.aml_env* file was created in {dotenv_path}")
        except IOError:
            raise IOError("The *.aml_env* file could not be created "
                          "- Please run 'workspace_setup()' again")

    return dotenv_path


def get_or_create_workspace(dotenv_file_path: str) -> Workspace:
    """
    Creates a new or retrieves an existing workspace

    :param dotenv_file_path: Full path of .aml_env file
    :return: Workspace object
    """

    # Extract variable values
    load_dotenv(dotenv_file_path)

    # Assign these values to our workspace variables
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")
    workspace_region = os.getenv("WORKSPACE_REGION")

    try:
        # Load the workspace
        if os.path.exists('./aml_config/config.json'):
            # From a configuration file
            ws = Workspace.from_config()

            ws_parameters = [ws.subscription_id, ws.resource_group,
                             ws.location, ws.name]
            dotenv_parameters = [subscription_id, resource_group,
                                 workspace_region, workspace_name]

            # Find potential differences between .aml_env and config.json files
            mask = [x != y for (x, y) in zip(dotenv_parameters, ws_parameters)]
            diff = list(compress(dotenv_parameters, mask))

            if diff:
                print(f" >>> Caution: *.aml_env* and *config.json* "
                      f"contents differ on {diff}<<<")
                choice = input(f"Please enter:\n"
                               f"1: To overwrite your *.aml_env* file "
                               f"(most common)\n"
                               f"2: To retrieve a different workspace "
                               f"from *{ws.name}*\n"
                               f"3: To create a new workspace\n")
                if choice == "1":
                    get_or_create_dot_env_file(
                        subscription_id=ws.subscription_id,
                        resource_group=ws.resource_group,
                        workspace_name=ws.name,
                        workspace_region=ws.location,
                        overwrite=True)
                    update_environment_variables(ws_parameters)
                elif choice == "2":
                    ws = Workspace(subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   workspace_name=workspace_name)
                    os.remove('./aml_config/config.json')
                    ws.write_config()
                else:
                    raise Exception
        else:
            # Or directly from Azure
            ws = Workspace(subscription_id=subscription_id,
                           resource_group=resource_group,
                           workspace_name=workspace_name)
            # And generate a local configuration file
            ws.write_config()
        print(f"Workspace *{workspace_name}* configuration was "
              f"successfully retrieved")
    except Exception:
        # Create a workspace from scratch
        print(f"Creating workspace *{workspace_name}* from scratch ...")
        ws = Workspace.create(name=workspace_name,
                              subscription_id=subscription_id,
                              resource_group=resource_group,
                              create_resource_group=True,
                              location=workspace_region
                              )
        ws.write_config()
        print(f"Workspace *{workspace_name}* has been successfully created.")

    return ws


def setup_workspace() -> Workspace:
    """
    Retrieves the workspace parameters
    and creates/retrieves the workspace in question
    Note: Both .aml_env and config.json can be useful,
    especially when notebooks get shared between users,
    who each have their own .aml_env files
    but are working on different subscriptions
    and/or different workspaces
    :return: (workspace object) Workspace to be used for work on Azure
    """

    path_to_dotenv_file = get_or_create_dot_env_file()

    return get_or_create_workspace(path_to_dotenv_file)
