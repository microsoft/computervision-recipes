#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>
#
# # Setup of an Azure workspace

# ## 1. Introduction <a id="intro"></a>
#
# This notebook is the first of a series (starting with "2x_") that leverage the [Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml). Azure ML, as we also call it, is a service that allows us to train, deploy, automate, and manage machine learning models, at scale, in the cloud.
#
# In this tutorial, we will set up an Azure ML workspace. Such resource organizes and coordinates the actions of many other Azure resources to assist in executing and sharing machine learning workflows. In particular, an Azure ML Workspace coordinates storage, databases, and compute resources providing added functionality for machine learning experimentation, deployment, inferencing, and the monitoring of deployed models.
#
# ## 2. Pre-requisites
# <a id="pre-reqs"></a>
#
# For this and all the other "2x_" notebooks to run properly on our machine, we need the following:
#
# * Local machine
#   * Any operation we will run should happen in the "cvbp" conda environment. [These instructions](https://github.com/Microsoft/ComputerVision/blob/master/classification/README.md#getting-started) explain how to do that.
#
# * Azure subscription
#   * We also need access to the Azure platform. Unless we already have one, we first should:
#     * [Create an account](https://azure.microsoft.com/en-us/free/services/machine-learning/)
#     * [Create a resource group and a workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace#portal).

# ## 3. Azure workspace <a id="workspace"></a>
#
# In the different tutorials present in this repository, we typically use the Azure ML SDK. It allows us to access our Azure resources programmatically. As we are running our notebooks in the "cvbp" conda environment, the SDK should already be installed on our machine. Let's check which version of the Azure SDK we are working with.

# In[ ]:


# For automatic reloading of modified libraries
get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# Azure
import azureml.core
from azureml.core import Workspace
from azureml.exceptions import ProjectSystemException, UserErrorException

# Check core SDK version number
print(f"Azure ML SDK Version: {azureml.core.VERSION}")


# We are now ready to load an existing workspace or create a new one, and save it to a local configuration file (`./aml_config/config.json`).
#
# If it is the first time we create a workspace, or if we are missing our `config.json` file, we need to provide the appropriate:
# - <b>subscription ID:</b> the ID of the Azure subscription we are using
# - <b>resource group:</b> the name of the resource group in which our workspace resides
# - <b>workspace region:</b> the geographical area in which our workspace resides (examples are available [here](https://azure.microsoft.com/en-us/global-infrastructure/geographies/))
# - <b>workspace name:</b> the name of the workspace we want to create or retrieve.

# In[ ]:


# Let's define these variables here - These pieces of information can be found on the portal
subscription_id = os.getenv("SUBSCRIPTION_ID", default="<our_subscription_id>")
resource_group = os.getenv("RESOURCE_GROUP", default="<our_resource_group>")
workspace_name = os.getenv("WORKSPACE_NAME", default="<our_workspace_name>")
workspace_region = os.getenv(
    "WORKSPACE_REGION", default="<our_workspace_region>"
)

try:
    # Let's load the workspace from the configuration file
    ws = Workspace.from_config()
    print("Workspace was loaded successfully from the configuration file")
except (UserErrorException, ProjectSystemException):
    # or directly from Azure, if it already exists (exist_ok=True).
    # If it does not exist, let's create a workspace from scratch
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=workspace_region,
        create_resource_group=True,
        exist_ok=True,
    )
    ws.write_config()
    print("Workspace was loaded successfully from Azure")


# Let's check that the workspace is properly loaded

# In[ ]:


# Print the workspace attributes
print(
    f"Workspace name: {ws.name}\n       Azure region: {ws.location}\n       Subscription id: {ws.subscription_id}\n       Resource group: {ws.resource_group}"
)


# We can also see this workspace on the Azure portal by sequentially clicking on:
# - Resource groups, and clicking the one we referenced above

# <img src="media/resource_group.jpg" width="800" alt="Azure portal view of resource group">

# - Workspace_name

# <img src="media/workspace.jpg" width="800" alt="Azure portal view of workspace">

# For more details on the setup of a workspace and other Azure resources, we can refer to this [configuration](https://github.com/Azure/MachineLearningNotebooks/blob/dcce6f227f9ca62e4c201fb48ae9dc8739eaedf7/configuration.ipynb) notebook.
#
# ## 4. Next steps <a id="next-step"></a>
#
# In this notebook, we loaded or created a new workspace, and stored configuration information in a `./aml_config/config.json` file. This is the file we will use in all the Azure ML-related notebooks in this repository. There, we will only need `ws = Workspace.from_config()`.
#
# In the next notebook, we will learn how to deploy a trained model as a web service on Azure.
