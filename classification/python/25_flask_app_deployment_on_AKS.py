#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>
#
#
# # Deployment of containerized Flask Application on AKS
#
# ## Table of content
# 1. [Introduction](#intro)
# 1. [Setup](#setup)
# 1. [Registration on Azure Container Registry](#register)
# 1. [Installation and configuration of kubectl](#kube)
# 1. [Creation of a static public IP address](#static)
# 1. [Deployment of our registered Docker container on AKS](#deploy)
# 1. [Website testing](#web_test)
# 1. [Service monitoring](#insights)
# 1. [Clean up](#clean)
#   1. [Static IP address detachment/deletion](#del_ip)
#   1. [Flask app service and deployment deletion](#del_svc)
#   1. [Application Insights deactivation and web service termination](#del_app_insights)
#   1. [Web service Docker image deletion](#del_image)
# 1. [Resources](#resources)

# ## 1. Introduction <a id="intro"/>
# In the prior notebook, we created a Flask application that allowed us to upload images and get their predicted classes and probabilities. We also packaged our app into a Docker container that we ran on our local machine.
#
# In this tutorial, we will learn how to:
# - Register our Docker image on the Azure Container Registry (ACR)
# - Deploy this Docker image, and consequently our Flask app, on the same AKS cluster as our existing web service.
#
# This will ultimately make our application accessible through a website, from any machine, and by more than one person.
#
# We are assuming here that we already have:
# - An Azure workspace
# - A web service that serves our machine learning model on AKS -- If that is not the case, we can refer to the [AKS deployment notebook](https://github.com/microsoft/ComputerVision/blob/master/classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb) to create one
# - A Flask application that provides us with a user interface, and calls our web service
# - A Docker image that contains our Flask application.
#
# If we are missing the last 2 objects, we can create them by following the steps shown in the [previous](https://github.com/microsoft/ComputerVision/blob/master/classification/notebooks/24_web_service_call_through_user_interface.ipynb) notebook.
#
# We will also use the Azure Command Line Interface (CLI), which installation instructions are provided  [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest).

# ## 2. Setup <a id="setup"/>
#
# To help with the registration and deployment of our new Docker image, we will need to access the resources associated with our Azure workspace. So, let's first retrieve latter.

# In[ ]:


# Import python and Azure library
import os

from azureml.core import Workspace

# Retrieve the workspace object
ws = Workspace.from_config()

# Print the workspace attributes
print(
    "Workspace name: " + ws.name,
    "Azure region: " + ws.location,
    "Subscription id: " + ws.subscription_id,
    "Resource group: " + ws.resource_group,
    sep="\n",
)


# Let's then move to the `flask_app/` directory. This folder contains all the files we have needed and will need for the creation of the Flask app, its deployment into a Docker image, and the deployment of that image onto AKS.

# In[ ]:


get_ipython().run_line_magic("cd", "flask_app")


# ## 3. Registration on Azure Container Registry <a id="register">
#
# Our Flask application can currently be accessed through our local Docker image. Now, our goal is to access it through a website. For this, we first need to register the Docker image into our [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-intro). This registry, associated with our workspace, already contains the Docker image we created in the [AKS deployment](https://github.com/Microsoft/ComputerVision/blob/master/classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb) notebook, which hosts our image classifier model.
#
# In the cells below, we will use the Azure CLI. As each of us has their own Azure resources, we will need to replace the variables identified by angled brackets (e.g. < subscription_id >) by our own values.
#
# *<b>Note:</b> For most users, the commands will follow the `az <action>` format. For others, `az.cmd <action>` may work better.*
#
# Let's first start by checking our Azure CLI version. Ideally, it should be 2.0.59 or later.

# In[ ]:


get_ipython().system('az --version | grep "azure-cli"')


# We also need to ensure that we use the right Azure subscription ID. Let's set it up properly.

# In[ ]:


print(f"My subscription ID is {ws.subscription_id}")


# In[ ]:


get_ipython().system("az account set -s <subscription_ID>")
# Set the account used by the CLI to the one associated with this <subscription ID>
# Let's replace <subscription_ID> by the result we obtained in the prior cell


# We also need to know the name of our container registry

# In[ ]:


acr_name = os.path.basename(ws.get_details()["containerRegistry"])
print(f"My Azure Container Registry is *{acr_name}*")


# In[ ]:


get_ipython().system("az acr login --name <acr_name>")
# Let's replace <acr_name> by the value we obtained above
# This takes a few seconds


# At this point, we may be asked to enter a username and a password. If that is the case, we need to:
# - Go to the Azure portal
# - Navigate to our resource group
# - Click on our workspace name
# - Click on the link next to "Registry" on the upper right of the screen - This takes us to our container registry
# - Click on "Access keys" on the menu
# - Copy and paste the username and one of the passwords available on that page.
#
# After providing these credentials, we should see `Login succeeded` in our notebook.
#
# We have just established a connection to our Azure remote resources. Let's now tag our local image, so we can recognize it later, and register it into our ACR.

# In[ ]:


get_ipython().system(
    "docker tag flaskappdockerimage <acr_name>.azurecr.io/flaskappdockerimage:v1"
)


# Let's check that the tag was applied properly

# In[ ]:


get_ipython().system(
    'docker images --filter=reference="flaskappdockerimage:*"'
)


# We are now ready to register our image.
#
# <i><b>Note:</b> This takes a few minutes.</i>

# In[ ]:


get_ipython().system(
    "docker push <acr_name>.azurecr.io/flaskappdockerimage:v1"
)


# We can check that this operation completed by going to the Azure portal. Still in the Registry section, where we found the credentials above, let's click on "Repositories". There, we should see our newly registered image.
#
# <img src="media/acr_repositories.jpg" width="700" align="left">

# By clicking on it, we should see the "v1" tag we just added.
#
# <img src="media/acr_tag.jpg" width="700" align="left">

# Clicking on this tag, shows us more details about the image, in particular what we did to create it, with the contents of the Dockerfile.
# <img src="media/acr_manifest.jpg" width="700" align="left">

# We can also query the ACR and extract the list of images it currently contains. Both `image-classif-resnet18-f48`, which contains our image classifier model, and `flaskappdockerimage` are there.

# In[ ]:


get_ipython().system("az acr repository list --name <acr_name> --output table")


# In[ ]:


get_ipython().system(
    "az acr repository show-tags --name <acr_name> --repository flaskappdockerimage --output table"
)
# Tag "v1", as created above


# The "Overview" section of the Azure Container Registry also shows a spike in activity. This corresponds to the moment when we registered our `flaskappdockerimage`.
#
# <img src="media/acr_activity_spike.jpg" width="300" align="left">

# ## 4. Installation and configuration of kubectl <a id="kube">
# To continue our deployment on Azure Kubernetes Service, we now need to use the Kubernetes CLI: kubectl. If we don't already have it, let's install it.

# In[ ]:


get_ipython().system("az aks install-cli")


# This should display a message similar to the following one, inviting us to add information to our PATH environment variable:
# ```
# Add kubectl.exe to Path environment variable
# Please add "<home_path>\.azure-kubectl" to your search PATH so the kubectl.exe can be found. 2 options:
#     1. Run "set PATH=%PATH%;<home_path>\.azure-kubectl" or "$env:path += '<home_path>\.azure-kubectl'" for PowerShell.
#     This is good for the current command session.
#     2. Update system PATH environment variable by following "Control Panel->System->Advanced->Environment Variables", and re-open the command window. You only need to do it once
# ```
# The message will differ depending on the operating system we are using.

# Let's also check that we are still using the right subscription id, and remind ourselves of our resource group, if we forgot about it.

# In[ ]:


# Check the subscription ID used
get_ipython().system("az account show")

# If it is not the right one, we can set it again
# !az account set -s <subscription ID>


# In[ ]:


print(f"My resource group is {ws.resource_group}")


# Let's now configure kubectl to connect to our AKS cluster. Here, we will use its full name.

# In[ ]:


full_aks_cluster_name = os.path.basename(
    ws.compute_targets["imgclass-aks-cpu"].cluster_resource_id
)
print(f"My AKS cluster name is {full_aks_cluster_name}")


# In[ ]:


get_ipython().system(
    "az aks get-credentials --resource-group <resource_group> --name <full_aks_cluster_name>"
)


# The command above gets credentials for our AKS cluster in our resource group. It also creates a `~/.kube/config` file, which contains information on our AKS cluster and users.
#
# Let's verify that our connection was properly set up

# In[ ]:


get_ipython().system("kubectl get nodes")


# We should see something silimar to the following:
#
# ```
# NAME                       STATUS   ROLES   AGE     VERSION
# aks-agentpool-xxxxxxxx-0   Ready     agent     1d        v1.12.7
# aks-agentpool-xxxxxxxx-1   Ready     agent     1d        v1.12.7
# aks-agentpool-xxxxxxxx-2   Ready     agent     1d        v1.12.7
# ```
#
# The connection succeeded, as we can get information on our AKS cluster from within our notebook. Here, we see that it is composed of one pool of 3 virtual machines.

# ## 5. Creation of a static public IP address <a id="static">
#
# When we [created](https://github.com/Microsoft/ComputerVision/blob/master/classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb) our AKS cluster, we set it up, by default, only with an internal IP address, to allow only communication [within the cluster](https://docs.microsoft.com/en-us/azure/aks/concepts-network).
#
# Our goal here, however, is to have a public URL that can be used to call our service externally from the cluster. We then need to create a [static public IP address](https://docs.microsoft.com/en-us/azure/aks/static-ip). Such an IP address should be created in the AKS node resource group, so all resources are grouped together. This requires us to know what resource group our AKS cluster belongs to. Let's extract it first.

# In[ ]:


get_ipython().system(
    "az aks show --resource-group <resource_group> --name <full_aks_cluster_name> --query nodeResourceGroup -o tsv"
)
# Here <resource_group> is the one in which our workspace is (i.e. the output of ws.resource_group)


# The command above returns the <b>node</b> resource group for our AKS cluster. It is typically of the following format:
# ```
# MC_<resource_group>_<full_aks_cluster_name>_<workspace_region>
# ```
#
# This string is the resource group we will use below.
#
# Let's now create our IP address and give it the name `ourAKSPublicIP`.

# In[ ]:


get_ipython().system(
    "az network public-ip create --resource-group <node_resource_group> --name ourAKSPublicIP --allocation-method=static"
)
# Let's replace <node_resource_group> by the result from the prior command


# This should return a JSON object with the following shape:
#
# ```{
#   "publicIp": {
#     "dnsSettings": null,
#     "etag": "<unique_identifier>",
#     "id": "/subscriptions/<subscription_id>/resourceGroups/<node_resource_group>/providers/Microsoft.Network/publicIPAddresses/ourAKSPublicIP",
#     "idleTimeoutInMinutes": 4,
#     "ipAddress": "xxx.xxx.xxx.xx",
#     "name": "ourAKSPublicIP",
#     [...]
#   }
# }
# ```
#
# To make sure that this step worked properly, let's query the IP address we just created. For this, we need the name of our <b>node</b> resource group and of our IP address (here `ourAKSPublicIP`).

# In[ ]:


get_ipython().system(
    "az network public-ip show --resource-group <node_resource_group> --name ourAKSPublicIP --query ipAddress --output tsv"
)


# This should return the same IP address as the one contained in the JSON object we obtained above.
#
# We can also check for the presence of our new IP address on the Azure portal. In the general "Resource Group" section of the portal, we can search for our node resource group. After clicking on the returned result, we can see our IP address present in the list of created resources.
#
# <img src="media/node_resource_group.jpg" width="600" align="left">

# By clicking on it, we have access to logs, configuration and other properties of our IP address.
# <img src="media/ip_address.jpg" width="600" align="left">

# ## 6. Deployment of our registered Docker container on AKS <a id="deploy">
#
# The last file we will use in this project is `aks_deployment.yaml`. It specifies how to create a new deployment for our Flask app to run on (e.g. Docker image used, port, number of replicas, IP address, etc.).
#
# Let's open this file and replace `<static_ip_address>` and `<acr_name>` by the address we just obtained and by the name of our Azure Container Registry. All the pieces we need for the deployment on AKS are now ready. The `kubectl apply` command will now parse this yaml file (also called a `manifest`) and create the `Deployment` and `Service` objects we defined there.
#
# <b>Note:</b> If we use Windows, let's make sure to run `dos2unix.exe aks_deployment.yaml`, so the next command can run successfully.

# In[ ]:


get_ipython().system("kubectl apply -f aks_deployment.yaml")


# This should display the following results:
#
# ```
# deployment.apps "flask-app-deployment" created
# service "flask-app-deployment" created
# ```
#
#
# When the application runs, a Kubernetes service exposes the application to the internet. This process can take <b>a few minutes</b> to complete. We can monitor progress of the service deployment by using the `kubectl get service --watch` command. Better yet is to use `kubectl get all`. This shows all AKS pods, replicas, deployments and services on our AKS cluster. If we run it several times, a few minutes apart, we should see a change in the "EXTERNAL-IP" column for the deployed services. It should change from "Pending" to the IP address we just created. Additionally, the "STATUS" column of the pods that run our Flask app should change from "ContainerCreating" to "Pending" to "Running".

# In[ ]:


get_ipython().system("kubectl get all")
# Let's run this command 2 or 3 times, a few minutes apart


# ##### Debugging
#
# If there is any issue with deployment, especially the above command showing something different from "ContainerCreating", "Pending" or "Running", we can delete our `Deployment` and `Service` objects, fix the problem in our `aks_deployment.yaml` file, and run the `kubectl apply` command again.

# In[ ]:


# !kubectl delete svc flask-app-deployment
# !kubectl delete deploy flask-app-deployment
# # Fix aks_deployment.yaml
# !kubectl apply -f aks_deployment.yaml


# ## 7. Website testing <a id="web_test">

# By now, our Flask application should be up and running. Let's test it.
#
# Let's copy our static IP address, followed by ":5000", into a browser (e.g. `http://xxx.xxx.xxx.xx:5000`)
#
# Here is our app website!
#
# <img src="media/website_ui.jpg" width="400" align="left">

# Let's now upload a few images. This should return a table with the images we uploaded, and their predicted classes and probabilities.
#
# <i><b>Note:</b>  The first time the service runs, it needs to be "primed", so it may take a little more time than for subsequent requests.</i>
#
# Thanks to our public IP address, we can now call our image classifier model, through our Flask application, from any computer.

# ## 8. Service monitoring <a id="insights"/>
#
# As before, we can track our web service health through the Application Insights service on the Azure portal (cf. [web service testing](https://github.com/microsoft/ComputerVision/blob/master/classification/notebooks/23_aci_aks_web_service.ipynb) notebook for details).

# ## 9. Clean up <a id="clean">
#
# In a real-life scenario, it is likely that the 2 services (i.e. image classifier and Flask application) we created would need to be up and running at all times. However, in the present demonstrative case, and now that we have verified that our services work, we can delete them as well as all the resources we used.
#
# Overall, with a workspace and a web service running on a CPU-based AKS cluster, we incurred a cost of about $15 a day. To get a better sense of pricing, we can refer to [this calculator](https://azure.microsoft.com/en-us/pricing/calculator/?service=virtual-machines). We can also navigate to the [Cost Management + Billing pane](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/ModernBillingMenuBlade/BillingAccounts) on the portal, click on our subscription ID, and click on the Cost Analysis tab to check our credit usage.
#
# ### 9.A Static IP address detachment/deletion <a id="del_ip">
#
# We start with our static public IP address. As in the commands we used before, we need to replace `<resource_group>` by our own value.

# In[ ]:


get_ipython().system(
    "az network public-ip delete -g <resource_group> -n ourAKSPublicIP"
)


# ### 9.B Flask app service and deployment deletion <a id="del_svc">
#
# We then delete the service and deployment of our Flask application

# In[ ]:


# Flask app service deletion
get_ipython().system("kubectl delete svc flask-app-deployment")

# Flask app deployment deletion
get_ipython().system("kubectl delete deploy flask-app-deployment")


# ### 9.C Application Insights deactivation and web service termination <a id="del_app_insights">
#
# We can now delete resources associated with our image classifier web services.

# In[ ]:


# Telemetry deactivation
aks_service = ws.webservices["aks-cpu-image-classif-web-svc"]
aks_service.update(enable_app_insights=False)

# Services termination
aks_service.delete()

# Compute target deletion
aks_target = ws.compute_targets["imgclass-aks-cpu"]
aks_target.delete()


# ### 9.D Web service Docker image deletion <a id="del_image">
#
# Finally, we delete the Docker image that contains our image classifier model.

# In[ ]:


print("Docker images:")
for docker_im in ws.images:
    print(
        f"    --> Name: {ws.images[docker_im].name}\n    --> ID: {ws.images[docker_im].id}\n    --> Tags: {ws.images[docker_im].tags}\n    --> Creation time: {ws.images[docker_im].created_time}"
    )


# In[ ]:


docker_image = ws.images["image-classif-resnet18-f48"]
docker_image.delete()


# ## 10. Resources <a id="resources">
#
# Throughout this notebook, we used a variety of tools and concepts. If that is of interest, here a few resources we recommend reading:
# - Docker: [General documentation](https://docs.docker.com/get-started/), [CLI functions](https://docs.docker.com/engine/reference/commandline/docker/)
# - [Kubernetes](https://kubernetes.io/docs/home/)
# - [Another example of application deployment on AKS](https://docs.microsoft.com/en-us/azure/aks/tutorial-kubernetes-prepare-app)
# - IP address management:
#   - [Using Azure CLI](https://docs.microsoft.com/en-us/cli/azure/network/public-ip?view=azure-cli-latest)
#   - [From the Azure portal](https://docs.microsoft.com/en-us/azure/virtual-network/virtual-network-public-ip-address)
