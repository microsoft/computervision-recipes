# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import yaml

from azureml.core.conda_dependencies import CondaDependencies


def generate_yaml(
    directory: str,
    ref_filename: str,
    needed_libraries: list,
    conda_filename: str,
):
    """
    Creates a deployment-specific yaml file as a subset of
    the image classification environment.yml

    Also adds extra libraries, if not present in environment.yml

    Args:
        directory (string): Directory name of reference yaml file
        ref_filename (string): Name of reference yaml file
        needed_libraries (list of strings): List of libraries needed
        in the Docker container
        conda_filename (string): Name of yaml file to be deployed
        in the Docker container

    Returns: Nothing

    """

    with open(os.path.join(directory, ref_filename), "r") as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)

    # Extract libraries to be installed using conda
    extracted_libraries = [
        depend
        for depend in yaml_content["dependencies"]
        if any(lib in depend for lib in needed_libraries)
    ]

    # Extract libraries to be installed using pip
    if any(isinstance(x, dict) for x in yaml_content["dependencies"]):
        # if the reference yaml file contains a "pip" section,
        # find where it is in the list of dependencies
        ind = [
            yaml_content["dependencies"].index(depend)
            for depend in yaml_content["dependencies"]
            if isinstance(depend, dict)
        ][0]
        extracted_libraries += [
            depend
            for depend in yaml_content["dependencies"][ind]["pip"]
            if any(lib in depend for lib in needed_libraries)
        ]

    # Check whether additional libraries are needed
    not_found = [
        lib
        for lib in needed_libraries
        if not any(lib in ext for ext in extracted_libraries)
    ]

    # Create the deployment-specific yaml file
    conda_env = CondaDependencies()
    for ch in yaml_content["channels"]:
        conda_env.add_channel(ch)
    for library in extracted_libraries + not_found:
        conda_env.add_conda_package(library)

    # Display the environment
    print(conda_env.serialize_to_string())

    # Save the file to disk
    conda_env.save_to_file(
        base_directory=os.getcwd(), conda_file_path=conda_filename
    )

    # Note: For users interested in creating their own environments,
    # the only commands needed are:
    # conda_env = CondaDependencies()
    # conda_env.add_channel()
    # conda_env.add_conda_package()
    # conda_env.save_to_file()
