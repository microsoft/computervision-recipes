# python regular libraries
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path

# Azure
from azureml.core import Workspace


def workspace_parameters_input() -> tuple:
    """
    Turns on the interactive mode for the user to enter
    their subscription, resource group and workspace information

    :return: Tuple of information entered by the user
    """
    subscription_id = input("  > Your subscription ID: ")
    resource_gp = input("  > Your resource group: ")
    workspace_name = input("  > The name of your workspace: ")
    workspace_region = input("  > The region of your workspace\n"
                             "(full list available at "
                             "https://azure.microsoft.com/en-us/"
                             "global-infrastructure/geographies/): ")
    return subscription_id, resource_gp, workspace_name, workspace_region


def get_or_create_dot_env_file(subscription_id: str = None,
                               resource_gp: str = None,
                               workspace_name: str = None,
                               workspace_region: str = None,
                               override: bool = False) -> str:
    """
    Creates a .env file if it cannot find any
    in the current directory's parent hierarchy

    :param subscription_id: Subscription ID (available on Azure portal)
    :param resource_gp: Resource group (created by the user)
    :param workspace_name: Workspace name (chosen by the user)
    :param workspace_region: Workspace region (e.g. "westus", "eastus2, etc.)
    :param override: Whether file content should be overwritten or not
    :return: (string) Full path of the .env file
    """
    # Look for already existing .env file
    dotenv_path = find_dotenv()

    # If found and no need to overwrite it, retrieve it, else create a new one
    if dotenv_path != '' and not override:
        print("Found the *.env* file in: {}".format(dotenv_path))
    else:
        if not override:
            # if .env file not found, ask user to provide relevant information
            print("-- No *.env* file could be found - Let's create one --\n"
                  "-- Please provide (without quotes):")
            subscription_id, resource_gp, workspace_name, workspace_region = \
                workspace_parameters_input()

            # if .env file exists and needs to be overwritten,
            # use information from saved workspace

        home_dir = str(Path.home())
        env_file = open(os.path.join(home_dir, ".env"), "w")
        sentence = "SUBSCRIPTION_ID={}\nRESOURCE_GROUP={}\n" \
                   "WORKSPACE_NAME={}\nWORKSPACE_REGION={}"\
            .format(subscription_id, resource_gp,
                    workspace_name, workspace_region)
        env_file.write(sentence)
        env_file.close()

        dotenv_path = find_dotenv()
        if dotenv_path != '':
            print("The *.env* file was created in {}".format(dotenv_path))
        else:
            print("The *.env* file could not be created "
                  "- Please run 'workspace_setup()' again")

    return dotenv_path


def get_or_create_workspace(dotenv_file_path: str) -> Workspace:
    """
    Creates a new or retrieves an existing workspace

    :param dotenv_file_path: Full path of .env file
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

            if ws.subscription_id != subscription_id or \
                    ws.resource_group != resource_group or \
                    ws.name != workspace_name:
                print(" >>> Caution: *.env* and *config.json* "
                      "contents differ <<<")
                choice = input("Please enter:\n"
                               "1: To overwrite your *.env* file\n"
                               "2: To retrieve a different workspace from {}\n"
                               "3: To create a new workspace\n"
                               .format(ws.name))
                if choice == "1":
                    get_or_create_dot_env_file(
                        subscription_id=ws.subscription_id,
                        resource_gp=ws.resource_group,
                        workspace_name=ws.name,
                        workspace_region=ws.location,
                        override=True)
                elif choice == "2":
                    ws = Workspace(subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   workspace_name=workspace_name)
                    os.remove('./aml_config/config.json')
                    ws.write_config()
                else:
                    raise IOError
        else:
            # Or directly from Azure
            ws = Workspace(subscription_id=subscription_id,
                           resource_group=resource_group,
                           workspace_name=workspace_name)
            # And generate a local configuration file
            if os.path.exists('./aml_config/config.json'):
                print("Deleting the existing config.json file")
                os.remove('./aml_config/config.json')
            ws.write_config()
        print("Workspace *{}* configuration was successfully retrieved"
              .format(workspace_name))
    except IOError:
        # Create a workspace from scratch
        print("Creating workspace *{}* from scratch ..."
              .format(workspace_name))
        ws = Workspace.create(name=workspace_name,
                              subscription_id=subscription_id,
                              resource_group=resource_group,
                              create_resource_group=True,
                              location=workspace_region
                              )
        ws.write_config()
        print("Workspace *{}* has been successfully created."
              .format(workspace_name))

    return ws


def workspace_setup() -> Workspace:
    """
    Retrieves the workspace parameters
    and creates/retrieves the workspace in question
    Note: Both .env and config.json can be useful,
    especially when notebooks get shared between users,
    who each have their own .env files
    but are working on different subscriptions
    and/or different workspaces
    :return: (string) Full path of the .env file
    """

    path_to_dotenv_file = get_or_create_dot_env_file()
    work_space = get_or_create_workspace(path_to_dotenv_file)

    return work_space
