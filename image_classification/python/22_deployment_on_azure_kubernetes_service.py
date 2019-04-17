#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>
#
#
# # Deployment of a model as a service with Azure Kubernetes Service
#
# ## Table of contents
# 1. [Introduction](#intro)
# 1. [Pre-requisites](#pre-reqs)
# 1. [Library import](#libraries)
# 1. [Azure workspace](#workspace)
# 1. [Model deployment on AKS](#deploy)
#   1. [Docker image retrieval](#docker_image)
#   1. [AKS compute target creation](#compute)
#   1. [Monitoring activation](#monitor)
#   1. [Service deployment](#svc_deploy)
# 1. [Testing of the web service](#testing)
# 1. [Clean up](#clean)
#   1. [Application Insights deactivation](#insights)
#   1. [Service termination](#del_svc)
#   1. [Image deletion](#del_img)
#   1. [Workspace deletion](#del_workspace)
# 1. [Next steps](#next)
#
#
# ## 1. Introduction <a id="intro"/>
#
# In many real life scenarios, trained machine learning models need to be deployed to production. As we saw in the [first](https://github.com/Microsoft/ComputerVision/blob/staging/image_classification/notebooks/21_deployment_on_azure_container_instances.ipynb) deployment notebook, this can be done by deploying on Azure Container Instances. In this tutorial, we will get familiar with another way of implementing a model into a production environment, this time using [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/concepts-clusters-workloads) (AKS).
#
# AKS manages hosted Kubernetes environments. It makes it easy to deploy and manage containerized applications without container orchestration expertise. It also supports deployments with CPU clusters and deployments with GPU clusters. The latter have been shown to be [more economical and efficient](https://azure.microsoft.com/en-us/blog/gpus-vs-cpus-for-deployment-of-deep-learning-models/) when serving complex models such as deep neural networks, and/or when traffic to the web service is high (&gt; 100 requests/second).
#
#
# At the end of this tutorial, we will have learned how to:
#
# - Deploy a model as a web service using AKS
# - Monitor our new service.

# ## 2. Pre-requisites <a id="pre-reqs"/>
#
# This notebook relies on resources we created in [21_deployment_on_azure_container_instances.ipynb](https://github.com/Microsoft/ComputerVision/blob/staging/image_classification/notebooks/21_deployment_on_azure_container_instances.ipynb):
# - Our local conda environment and Azure Machine Learning workspace
# - The Docker image that contains the model and scoring script needed for the web service to work.
#
# If we are missing any of these, we should go back to the previous notebook and generate them.

# ## 3. Library import <a id="libraries"/>
#
# Now that our prior resources are available, let's first import a few libraries we will need for the deployment on AKS.

# In[1]:


# For automatic reloading of modified libraries
get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# Azure
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice, Webservice


# ## 4. Azure workspace <a id="workspace"/>
#
# In the prior notebook, we retrieved an existing or created a new workspace, and generated an `./aml_config/config.json` file.
# Let's use it to load this workspace.
#
# <i><b>Note:</b> The Docker image we will use below is attached to the workspace we used in the prior notebook. It is then important to use the same workspace here. If, for any reason, we need to use a separate workspace here, then the steps followed to create a Docker image containing our image classifier model in the prior notebook, should be reproduced here.</i>

# In[2]:


ws = Workspace.from_config()
# from_config() refers to this config.json file by default


# Let's check that the workspace is properly loaded

# In[3]:


# Print the workspace attributes
print(
    "Workspace name: " + ws.name,
    "Azure region: " + ws.location,
    "Subscription id: " + ws.subscription_id,
    "Resource group: " + ws.resource_group,
    sep="\n",
)


# ## 5. Model deployment on AKS <a id="deploy">
#
# ### 5.A Docker image retrieval <a id="docker_image">
#
# As for the deployment on Azure Container Instances, we will use Docker containers. The Docker image we created in the prior notebook is very much suitable for our deployment on Azure Kubernetes Service, as it contains the libraries we need and the model we registered. Let's make sure this Docker image is still available (if not, we can just run the cells of section "6. Model deployment on Azure" of the [prior notebook](https://github.com/Microsoft/ComputerVision/blob/staging/image_classification/notebooks/21_deployment_on_azure_container_instances.ipynb)).

# In[4]:


print("Docker images:")
for docker_im in ws.images:
    print(
        f" --> Name: {ws.images[docker_im].name}\n     --> ID: {ws.images[docker_im].id}\n     --> Tags: {ws.images[docker_im].tags}\n     --> Creation time: {ws.images[docker_im].created_time}\n"
    )


# As we did not delete it in the prior notebook, our Docker image is still present in our workspace. Let's retrieve it.

# In[5]:


docker_image = ws.images["image-classif-resnet18-f48"]


# We can also check that the model it contains is the one we registered and used during our deployment on ACI. In our case, the Docker image contains only 1 model, so taking the 0th element of the `docker_image.models` list returns our model.
#
# <i><b>Note:</b> We will not use the `registered_model` object anywhere here. We are running the next 2 cells just for verification purposes.</i>

# In[6]:


registered_model = docker_image.models[0]


# In[7]:


print(
    f"Existing model:\n --> Name: {registered_model.name}\n --> Version: {registered_model.version}\n --> ID: {registered_model.id} \n --> Creation time: {registered_model.created_time}\n --> URL: {registered_model.url}"
)


# ### 5.B AKS compute target creation<a id="compute"/>
#
# In the case of deployment on AKS, in addition to the Docker image, we need to define computational resources. This is typically a cluster of CPUs or a cluster of GPUs. If we already have a Kubernetes-managed cluster in our workspace, we can use it, otherwise, we can create a new one.
#
# <i><b>Note:</b> The name we give to our compute target must be between 2 and 16 characters long.</i>
#
# Let's first check what types of compute resources we have, if any

# In[8]:


print("List of compute resources associated with our workspace:")
for cp in ws.compute_targets:
    print(f"   --> {cp}: {ws.compute_targets[cp]}")


# #### 5.B.a Creation of a new AKS cluster
#
# In the case where we have no compute resource available, we can create a new one. For this, we can choose between a CPU-based or a GPU-based cluster of virtual machines. There is a [wide variety](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general) of machine types that can be used. In the present example, however, we will not need the fastest machines that exist nor the most memory optimized ones. We will use typical default machines:
# - [Standard D3 V2](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general#dv2-series):
#   - 4 CPUs
#   - 14 GB of memory
#   - This allows for at least 12 cores to operate, which is the [minimum needed](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#create-a-new-cluster) for such cluster.
# - [Standard NC6](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu):
#   - 1 GPU
#   - 12 GB of GPU memory
#   - These machines also have 6 CPUs and 56 GB of memory.
#
# <i><b>Notes:</b>
# - These are Azure-specific denominations
# - Information on optimized machines can be found [here](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general#other-sizes)
# - By default, 3 agents get provisioned on a new AKS cluster. When choosing a type of machine, this parameter (`agent_count`) may need to be changed such that `agent_count x cpu_count` &ge; `12` virtual CPUs
# - Additional considerations on deployments using GPUs are available [here](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#deployment-considerations)</i>
#
# Here, we will use a cluster of CPUs.

# In[9]:


# Declare the name of the cluster
virtual_machine_type = "cpu"
aks_name = f"imgclass-aks-{virtual_machine_type}"

# Define the type of virtual machines to use
if aks_name not in ws.compute_targets:
    if virtual_machine_type == "gpu":
        vm_size = "Standard_NC6"
    else:
        vm_size = "Standard_D3_v2"

print(f"Our AKS computer target's name is: {aks_name}")


# In[10]:


# Configure the cluster
# Use the default configuration (can also provide parameters to customize)
# Full list available at https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.akscompute?view=azure-ml-py#provisioning-configuration-agent-count-none--vm-size-none--ssl-cname-none--ssl-cert-pem-file-none--ssl-key-pem-file-none--location-none--vnet-resourcegroup-name-none--vnet-name-none--subnet-name-none--service-cidr-none--dns-service-ip-none--docker-bridge-cidr-none-
if aks_name not in ws.compute_targets:
    prov_config = AksCompute.provisioning_configuration(vm_size=vm_size)


# In[11]:


if aks_name not in ws.compute_targets:
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )
    aks_target.wait_for_completion(show_output=True)
else:
    # Retrieve the already existing cluster
    aks_target = ws.compute_targets[aks_name]
    print(f"We retrieved the {aks_target.name} AKS compute target")


# When the cluster deploys successfully, we typically see the following:
#
# ```
# Creating ...
# SucceededProvisioning operation finished, operation "Succeeded"
# ```
#
# This step typically takes several minutes to complete.

# #### 5.B.b Alternative: Attachment of an existing AKS cluster
#
# Within our overall subscription, we may already have created an AKS cluster. This cluster may not be visible when we run the `ws.compute_targets` command, though. This is because it is not attached to our present workspace. If we want to use that cluster instead, we need to attach it to our workspace, first. We can do this as follows:

# In[ ]:


# existing_aks_name = '<name_of_the_existing_detached_aks_cluster>'
# resource_id = '/subscriptions/<subscription_id/resourcegroups/<resource_group>/providers/Microsoft.ContainerService/managedClusters/<aks_cluster_full_name>'
# # <aks_cluster_full_name> can be found by clicking on the aks cluster, in the Azure portal, as the "Resource ID" string
# # <subscription_id> can be obtained through ws.subscription_id, and <resource_group> through ws.resource_group

# attach_config = AksCompute.attach_configuration(resource_id=resource_id)
# aks_target = ComputeTarget.attach(workspace=ws, name=existing_aks_name, attach_configuration=attach_config)
# aks_target.wait_for_completion(show_output = True)


# This compute target can be seen on the Azure portal, under the `Compute` tab.
#
# <img src="media/aks_compute_target_cpu.jpg" width="900">

# In[12]:


# Check provisioning status
print(
    f"The AKS compute target provisioning {aks_target.provisioning_state.lower()} -- There were '{aks_target.provisioning_errors}' errors"
)


# The set of resources we will use to deploy our web service on AKS is now provisioned and available.
#
# ### 5.C Monitoring activation <a id="monitor"/>
#
# Once our web app is up and running, it is very important to monitor it, and measure the amount of traffic it gets, how long it takes to respond, the type of exceptions that get raised, etc. We will do so through [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview), which is an application performance management service. To enable it on our soon-to-be-deployed web service, we first need to update our AKS configuration file:

# In[13]:


# Set the AKS web service configuration and add monitoring to it
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)


# ### 5.D Service deployment <a id="svc_deploy"/>
#
# We are now ready to deploy our web service. As in the [first](https://github.com/Microsoft/ComputerVision/blob/staging/image_classification/notebooks/21_deployment_on_azure_container_instances.ipynb) notebook, we will deploy from the Docker image. It indeed contains our image classifier model and the conda environment needed for the scoring script to work properly. The parameters to pass to the `Webservice.deploy_from_image()` command are similar to those used for the deployment on ACI. The only major difference is the compute target (`aks_target`), i.e. the CPU cluster we just spun up.

# In[14]:


if aks_target.provisioning_state == "Succeeded":
    aks_service_name = "aks-cpu-image-classif-web-svc"
    aks_service = Webservice.deploy_from_image(
        workspace=ws,
        name=aks_service_name,
        image=docker_image,
        deployment_config=aks_config,
        deployment_target=aks_target,
    )
    aks_service.wait_for_deployment(show_output=True)
    print(f"The web service is {aks_service.state}")
else:
    raise ValueError("The web service deployment failed.")


# When successful, we should see the following:
#
# ```
# Creating service
# Running ...
# SucceededAKS service creation operation finished, operation "Succeeded"
# The web service is Healthy
# ```
#
# This deployment takes a few minutes to finish.
#
# In the case where the deployment is not successful, we can look at the service logs to debug. [These instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-troubleshoot-deployment) can also be helpful.

# In[ ]:


# Access to the service logs
# print(aks_service.get_logs())


# The new deployment can be seen on the portal, under the Deployments tab.
#
# <img src="media/aks_webservice_cpu.jpg" width="900">

# Our web service is up, and is running on AKS. We can now proceed to testing it.

# ## 6. Testing of the web service <a id="testing"/>
#
# Such testing is a whole task of its own, so we separated it from this notebook. We provide all the needed steps in [23_web_service_testing.ipynb](https://github.com/Microsoft/ComputerVision/blob/service_deploy/image_classification/notebooks/deployment/23_web_service_testing.ipynb). There, we test our service:
# - From within our workspace (using `aks_service.run()`)
# - From outside our workspace (using `requests.post()`)
# - From a Flask app running on our local machine
# - From a Flask app deployed on the same AKS cluster as our web service.

# ## 7. Clean up <a id="clean">
#
# In a real-life scenario, it is likely that the service we created would need to be up and running at all times. However, in the present demonstrative case, and once we have verified that our service works, we can delete it as well as all the resources we used.
#
# In this notebook, the only resource we added to our subscription, in comparison to what we had at the end of the notebook on ACI deployment, is the AKS cluster. There is no fee for cluster management. The only components we are paying for are:
# - the cluster nodes
# - the managed OS disks.
#
# Here, we used Standard D3 V2 machines, which come with a temporary storage of 200 GB. Over the course of this tutorial (assuming ~ 1 hour), this added less than $1 to our bill. Now, it is important to understand that each hour during which the cluster is up gets billed, whether the web service is called or not. The same is true for the ACI and workspace we have been using until now.
#
# To get a better sense of pricing, we can refer to [this calculator](https://azure.microsoft.com/en-us/pricing/calculator/?service=kubernetes-service#kubernetes-service). We can also navigate to the [Cost Management + Billing pane](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/ModernBillingMenuBlade/Overview) on the portal, click on our subscription ID, and click on the Cost Analysis tab to check our credit usage.
#
# If we plan on no longer using this web service, we can turn monitoring off, and delete the service itself as well as the associated Docker image.

# ### 7.A Application Insights deactivation <a id="insights"/>

# In[ ]:


# aks_service.update(enable_app_insights=False)


# ### 7.B Service termination <a id="del_svc"/>

# In[ ]:


# aks_service.delete()


# ### 7.C Image deletion <a id="del_img"/>

# In[ ]:


# image.delete()


# ### 7.D Workspace deletion  <a id="del_workspace"/>
# If our goal is to continue using our workspace, we should keep it available. On the contrary, if we plan on no longer using it and its associated resources, we can delete it.
#
# <i><b>Note:</b> Deleting the workspace will delete all the experiments, outputs, models, Docker images, deployments, etc. that we created in that workspace.</i>

# In[ ]:


# ws.delete(delete_dependent_resources=True)
# This deletes our workspace, the container registry, the account storage, Application Insights and the key vault


# ## 8. Next steps  <a id="next"/>
# In the [next notebook](https://github.com/Microsoft/ComputerVision/blob/service_deploy/image_classification/notebooks/deployment/23_web_service_testing.ipynb), we will test the web services we deployed on ACI and on AKS. We will also learn how a Flask app, with an interactive user interface, can be used to call our web service.
