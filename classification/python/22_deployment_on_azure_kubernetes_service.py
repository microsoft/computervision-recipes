#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>
#
#
# # Deployment of a model to Azure Kubernetes Service (AKS)
#
# ## Table of contents
# 1. [Introduction](#intro)
# 1. [Model deployment on AKS](#deploy)
#   1. [Workspace retrieval](#workspace)
#   1. [Docker image retrieval](#docker_image)
#   1. [AKS compute target creation](#compute)
#   1. [Monitoring activation](#monitor)
#   1. [Service deployment](#svc_deploy)
# 1. [Clean up](#clean)
# 1. [Next steps](#next)
#
#
# ## 1. Introduction <a id="intro"/>
#
# In many real life scenarios, trained machine learning models need to be deployed to production. As we saw in the [prior](21_deployment_on_azure_container_instances.ipynb) deployment notebook, this can be done by deploying on Azure Container Instances. In this tutorial, we will get familiar with another way of implementing a model into a production environment, this time using [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/concepts-clusters-workloads) (AKS).
#
# AKS manages hosted Kubernetes environments. It makes it easy to deploy and manage containerized applications without container orchestration expertise. It also supports deployments with CPU clusters and deployments with GPU clusters.
#
# At the end of this tutorial, we will have learned how to:
#
# - Deploy a model as a web service using AKS
# - Monitor our new service.

# ### Pre-requisites <a id="pre-reqs"/>
#
# This notebook relies on resources we created in [21_deployment_on_azure_container_instances.ipynb](21_deployment_on_azure_container_instances.ipynb):
# - Our Azure Machine Learning workspace
# - The Docker image that contains the model and scoring script needed for the web service to work.
#
# If we are missing any of these, we should go back and run the steps from the sections "Pre-requisites" to "3.D Environment setup" to generate them.
#
# ### Library import <a id="libraries"/>
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


# ## 2. Model deployment on AKS <a id="deploy"/>
#
# ### 2.A Workspace retrieval <a id="workspace">
#
# Let's now load the workspace we used in the [prior notebook](21_deployment_on_azure_container_instances.ipynb).
#
# <i><b>Note:</b> The Docker image we will use below is attached to that workspace. It is then important to use the same workspace here. If, for any reason, we needed to use another workspace instead, we would need to reproduce, here, the steps followed to create a Docker image containing our image classifier model in the prior notebook.</i>

# In[2]:


ws = Workspace.setup()
# setup() refers to our config.json file by default

# Print the workspace attributes
print(
    "Workspace name: " + ws.name,
    "Workspace region: " + ws.location,
    "Subscription id: " + ws.subscription_id,
    "Resource group: " + ws.resource_group,
    sep="\n",
)


# ### 2.B Docker image retrieval <a id="docker_image">
#
# We can reuse the Docker image we created in section 3. of the [previous tutorial](21_deployment_on_azure_container_instances.ipynb). Let's make sure that it is still available.

# In[3]:


print("Docker images:")
for docker_im in ws.images:
    print(
        f" --> Name: {ws.images[docker_im].name}\n     --> ID: {ws.images[docker_im].id}\n     --> Tags: {ws.images[docker_im].tags}\n     --> Creation time: {ws.images[docker_im].created_time}\n"
    )


# As we did not delete it in the prior notebook, our Docker image is still present in our workspace. Let's retrieve it.

# In[4]:


docker_image = ws.images["image-classif-resnet18-f48"]


# We can also check that the model it contains is the one we registered and used during our deployment on ACI. In our case, the Docker image contains only 1 model, so taking the 0th element of the `docker_image.models` list returns our model.
#
# <i><b>Note:</b> We will not use the `registered_model` object anywhere here. We are running the next 2 cells just for verification purposes.</i>

# In[6]:


registered_model = docker_image.models[0]

print(
    f"Existing model:\n --> Name: {registered_model.name}\n --> Version: {registered_model.version}\n --> ID: {registered_model.id} \n --> Creation time: {registered_model.created_time}\n --> URL: {registered_model.url}"
)


# ### 2.C AKS compute target creation<a id="compute"/>
#
# In the case of deployment on AKS, in addition to the Docker image, we need to define computational resources. This is typically a cluster of CPUs or a cluster of GPUs. If we already have a Kubernetes-managed cluster in our workspace, we can use it, otherwise, we can create a new one.
#
# <i><b>Note:</b> The name we give to our compute target must be between 2 and 16 characters long.</i>
#
# Let's first check what types of compute resources we have, if any

# In[7]:


print("List of compute resources associated with our workspace:")
for cp in ws.compute_targets:
    print(f"   --> {cp}: {ws.compute_targets[cp]}")


# In the case where we have no compute resource available, we can create a new one. For this, we can choose between a CPU-based or a GPU-based cluster of virtual machines. The latter is typically better suited for web services with high traffic (i.e. &gt 100 requests per second) and high GPU utilization. There is a [wide variety](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general) of machine types that can be used. In the present example, however, we will not need the fastest machines that exist nor the most memory optimized ones. We will use typical default machines:
# - [Standard D3 V2](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general#dv2-series):
#   - 4 vCPUs
#   - 14 GB of memory
# - [Standard NC6](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu):
#   - 1 GPU
#   - 12 GB of GPU memory
#   - These machines also have 6 vCPUs and 56 GB of memory.
#
# <i><b>Notes:</b></i>
# - These are Azure-specific denominations
# - Information on optimized machines can be found [here](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-general#other-sizes)
# - When configuring the provisioning of an AKS cluster, we need to choose a type of machine, as examplified above. This choice must be such that the number of virtual machines (also called `agent nodes`), we require, multiplied by the number of vCPUs on each machine must be greater than or equal to 12 vCPUs. This is indeed the [minimum needed](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#create-a-new-aks-cluster) for such cluster. By default, a pool of 3 virtual machines gets provisioned on a new AKS cluster to allow for redundancy. So, if the type of virtual machine we choose has a number of vCPUs (`vm_size`) smaller than 4, we need to increase the number of machines (`agent_count`) such that `agent_count x vm_size` &ge; `12` virtual CPUs. `agent_count` and `vm_size` are both parameters we can pass to the `provisioning_configuration()` method below.
# - [This document](https://docs.microsoft.com/en-us/azure/templates/Microsoft.ContainerService/2019-02-01/managedClusters?toc=%2Fen-us%2Fazure%2Fazure-resource-manager%2Ftoc.json&bc=%2Fen-us%2Fazure%2Fbread%2Ftoc.json#managedclusteragentpoolprofile-object) provides the full list of virtual machine types that can be deployed in an AKS cluster
# - Additional considerations on deployments using GPUs are available [here](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#deployment-considerations)
# - If the Azure subscription we are using is shared with other users, we may encounter [quota restrictions](https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits) when trying to create a new cluster. To ensure that we have enough machines left, we can go to the Portal, click on our workspace name, and navigate to the `Usage + quotas` section. If we need more machines than are currently available, we can request a [quota increase](https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#request-quota-increases).
#
# Here, we will use a cluster of CPUs. The creation of such resource typically takes several minutes to complete.

# In[8]:


# Declare the name of the cluster
virtual_machine_type = "cpu"
aks_name = f"imgclass-aks-{virtual_machine_type}"

if aks_name not in ws.compute_targets:
    # Define the type of virtual machines to use
    if virtual_machine_type == "gpu":
        vm_size_name = "Standard_NC6"
    else:
        vm_size_name = "Standard_D3_v2"

    # Configure the cluster using the default configuration (i.e. with 3 virtual machines)
    prov_config = AksCompute.provisioning_configuration(
        vm_size=vm_size_name, agent_count=3
    )

    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )
    aks_target.wait_for_completion(show_output=True)
    print(f"We created the {aks_target.name} AKS compute target")
else:
    # Retrieve the already existing cluster
    aks_target = ws.compute_targets[aks_name]
    print(f"We retrieved the {aks_target.name} AKS compute target")


# If we need a more customized AKS cluster, we can provide more parameters to the `provisoning_configuration()` method, the full list of which is available [here](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.akscompute?view=azure-ml-py#provisioning-configuration-agent-count-none--vm-size-none--ssl-cname-none--ssl-cert-pem-file-none--ssl-key-pem-file-none--location-none--vnet-resourcegroup-name-none--vnet-name-none--subnet-name-none--service-cidr-none--dns-service-ip-none--docker-bridge-cidr-none-).
#
# When the cluster deploys successfully, we typically see the following:
#
# ```
# Creating ...
# SucceededProvisioning operation finished, operation "Succeeded"
# ```
#
# In the case when our cluster already exists, we get the following message:
#
# ```
# We retrieved the <aks_cluster_name> AKS compute target
# ```

# This compute target can be seen on the Azure portal, under the `Compute` tab.
#
# <img src="media/aks_compute_target_cpu.jpg" width="900">

# In[9]:


# Check provisioning status
print(
    f"The AKS compute target provisioning {aks_target.provisioning_state.lower()} -- There were '{aks_target.provisioning_errors}' errors"
)


# The set of resources we will use to deploy our web service on AKS is now provisioned and available.
#
# ### 2.D Monitoring activation <a id="monitor"/>
#
# Once our web app is up and running, it is very important to monitor it, and measure the amount of traffic it gets, how long it takes to respond, the type of exceptions that get raised, etc. We will do so through [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview), which is an application performance management service. To enable it on our soon-to-be-deployed web service, we first need to update our AKS configuration file:

# In[10]:


# Set the AKS web service configuration and add monitoring to it
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)


# ### 2.E Service deployment <a id="svc_deploy"/>
#
# We are now ready to deploy our web service. As in the [first](21_deployment_on_azure_container_instances.ipynb) notebook, we will deploy from the Docker image. It indeed contains our image classifier model and the conda environment needed for the scoring script to work properly. The parameters to pass to the `Webservice.deploy_from_image()` command are similar to those used for the deployment on ACI. The only major difference is the compute target (`aks_target`), i.e. the CPU cluster we just spun up.
#
# <i><b>Note:</b> This deployment takes a few minutes to complete.</i>

# In[11]:


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
# In the case where the deployment is not successful, we can look at the service logs to debug. [These instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-troubleshoot-deployment) can also be helpful.

# In[ ]:


# Access to the service logs
# print(aks_service.get_logs())


# The new deployment can be seen on the portal, under the Deployments tab.
#
# <img src="media/aks_webservice_cpu.jpg" width="900">

# Our web service is up, and is running on AKS.

# ## 3. Clean up <a id="clean">
#
# In a real-life scenario, it is likely that the service we created would need to be up and running at all times. However, in the present demonstrative case, and once we have verified that our service works (cf. "Next steps" section below), we can delete it as well as all the resources we used.
#
# In this notebook, the only resource we added to our subscription, in comparison to what we had at the end of the notebook on ACI deployment, is the AKS cluster. There is no fee for cluster management. The only components we are paying for are:
# - the cluster nodes
# - the managed OS disks.
#
# Here, we used Standard D3 V2 machines, which come with a temporary storage of 200 GB. Over the course of this tutorial (assuming ~ 1 hour), this changed almost nothing to our bill. Now, it is important to understand that each hour during which the cluster is up gets billed, whether the web service is called or not. The same is true for the ACI and workspace we have been using until now.
#
# To get a better sense of pricing, we can refer to [this calculator](https://azure.microsoft.com/en-us/pricing/calculator/?service=kubernetes-service#kubernetes-service). We can also navigate to the [Cost Management + Billing pane](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/ModernBillingMenuBlade/Overview) on the portal, click on our subscription ID, and click on the Cost Analysis tab to check our credit usage.
#
# If we plan on no longer using this web service, we can turn monitoring off, and delete the compute target, the service itself as well as the associated Docker image.

# In[ ]:


# Application Insights deactivation
# aks_service.update(enable_app_insights=False)

# Service termination
# aks_service.delete()

# Compute target deletion
# aks_target.delete()
# This command executes fast but the actual deletion of the AKS cluster takes several minutes

# Docker image deletion
# docker_image.delete()


# At this point, all the service resources we used in this notebook have been deleted. We are only now paying for our workspace.
#
# If our goal is to continue using our workspace, we should keep it available. On the contrary, if we plan on no longer using it and its associated resources, we can delete it.
#
# <i><b>Note:</b> Deleting the workspace will delete all the experiments, outputs, models, Docker images, deployments, etc. that we created in that workspace.</i>

# In[ ]:


# ws.delete(delete_dependent_resources=True)
# This deletes our workspace, the container registry, the account storage, Application Insights and the key vault


# ## 4. Next steps  <a id="next"/>
# In the [next notebook](23_aci_aks_web_service_testing.ipynb), we will test the web services we deployed on ACI and on AKS.
