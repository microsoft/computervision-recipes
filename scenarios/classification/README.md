# Image classification

This directory provides examples and best practices for building image classification systems. Our goal is to enable users to easily and quickly train high-accuracy classifiers on their own datasets. We provide example notebooks with pre-set default parameters that are shown to work well on a variety of data sets. We also include extensive documentation of common pitfalls and best practices. Additionally, we show how Azure, Microsoft's cloud computing platform, can be used to speed up training on large data sets or deploy models as web services.

We recommend using PyTorch as a Deep Learning platform for its ease of use, simplicity when debugging, and popularity in the data science community. For Computer Vision functionality, we also rely heavily on [fast.ai](https://github.com/fastai/fastai), a PyTorch data science library which comes with rich deep learning features and extensive documentation. We highly recommend watching the [2019 fast.ai lecture series](https://course.fast.ai/videos/?lesson=1) video to understand the underlying technology. Fast.ai's [documentation](https://docs.fast.ai/) is also a valuable resource.


## Frequently asked questions

Answers to Frequently Asked Questions such as "How many images do I need to train a model?" or "How to annotate images?" can be found in the [FAQ.md](FAQ.md) file.


## Notebooks

We provide several notebooks to show how image classification algorithms are designed, evaluated and operationalized. Notebooks starting with `0` are intended to be run sequentially, as there are dependencies between them. These notebooks contain introductory "required" material. Notebooks starting with `1` can be considered optional and contain more complex and specialized topics.

While all notebooks can be executed in Windows, we have found that fast.ai is much faster on the Linux operating system. Additionally, using GPU dramatically improves training speeds. We suggest using an Azure Data Science Virtual Machine with V100 GPU ([instructions](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm), [price table](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/)). 

We have also found that some browsers do not render Jupyter widgets correctly. If you have issues, try using an alternative browser, such as  Edge or Chrome.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](00_webcam.ipynb)| Demonstrates inference on an image from your computer's webcam using a pre-trained model.
| [01_training_introduction.ipynb](01_training_introduction.ipynb)| Introduces some of the basic concepts around model training and evaluation.|
| [02_multilabel_classification.ipynb](02_multilabel_classification.ipynb)| Introduces multi-label classification and introduces key differences between training a multi-label and single-label classification models.|
| [03_training_accuracy_vs_speed.ipynb](03_training_accuracy_vs_speed.ipynb)| Trains a model with high accuracy vs one with a fast inferencing speed. *<font color="orange"> Use this to train on your own datasets! </font>* |
| [10_image_annotation.ipynb](10_image_annotation.ipynb)| A simple UI to annotate images. |
| [11_exploring_hyperparameters.ipynb](11_exploring_hyperparameters.ipynb)| Finds optimal model parameters using grid search. |
| [12_hard_negative_sampling.ipynb](12_hard_negative_sampling.ipynb)| Demonstrated how to use hard negatives to improve your model performance.  |
| [20_azure_workspace_setup.ipynb](20_azure_workspace_setup.ipynb)| Sets up your Azure resources and Azure Machine Learning workspace. |
| [21_deployment_on_azure_container_instances.ipynb](21_deployment_on_azure_container_instances.ipynb)| Deploys a trained model exposed on a REST API using Azure Container Instances (ACI). |
| [22_deployment_on_azure_kubernetes_service.ipynb](22_deployment_on_azure_kubernetes_service.ipynb)| Deploys a trained model exposed on a REST API using the Azure Kubernetes Service (AKS). |
| [23_aci_aks_web_service_testing.ipynb](23_aci_aks_web_service_testing.ipynb)| Tests the deployed models on either ACI or AKS. |
| [24_exploring_hyperparameters_on_azureml.ipynb](24_exploring_hyperparameters_on_azureml.ipynb)| Performs highly parallel parameter sweeping using AzureML's HyperDrive. |


## Using a Virtual Machine

You may want to use a virtual machine to run the notebooks. Doing so will give you a lot more flexibility -- whether it is using a GPU enabled machine or simply working in Linux. 

__Data Science Virtual Machine Builder__

One easy way to create your VM is to use the 'create_dsvm.py' tool located inside of the 'tools' folder in the root directory of the repo. Simply run `python tools/create_dsvm.py` at the root level of the repo. This tool preconfigures your virtual machine with the appropriate settings for working with this repository.

__Using the Azure Portal or CLI__

You can also spin up a VM directly using the Azure portal. For this repository,
you will want to create a Data Science Virtual Machine (DSVM). To do so, follow
[this](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)
link that shows you how to provision your VM through the portal.

You can alternatively use the Azure command line (CLI) as well. Follow
[this](https://docs.microsoft.com/en-us/cli/azure/azure-cli-vm-tutorial?view=azure-cli-latest)
link to learn more about the Azure CLI and how it can be used to provision
resources.

Once your virtual machine has been created, ssh and tunnel into the machine, then run the "Getting started" steps inside of it. The 'create_dsvm' tool will show you how to properly perform the tunneling too. If you created your virtual machine using the portal or the CLI, you can tunnel your jupyter notebook ports using the following command:
```
$ssh -L local_port:remote_address:remote_port username@server.com
```



## Azure-enhanced notebooks

Azure products and services are used in certain notebooks to enhance the efficiency of developing classification systems at scale.

To successfully run these notebooks, the users **need an Azure subscription** or can [use Azure for free](https://azure.microsoft.com/en-us/free/).

The Azure products featured in the notebooks include:

* [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) - Azure Machine Learning service is a cloud service used to train, deploy, automate, and manage machine learning models, all at the broad scale that the cloud provides. It is used across various notebooks for the AI model development related tasks such as deployment. [20_azure_workspace_setup](20_azure_workspace_setup.ipynb) shows how to set up your Azure resources and connect to an Azure Machine Learning service workspace.

* [Azure Container Instance](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#aci) - You can use Azure Machine Learning service to host your classification model in a web service deployment on Azure Container Instance (ACI). ACI is good for low scale, CPU-based workloads. [21_deployment_on_azure_container_instances](21_deployment_on_azure_container_instances.ipynb) explains how to deploy a web service to ACI through Azure ML.

* [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#aks) - You can use Azure Machine Learning service to host your classification model in a web service deployment on Azure Kubernetes Service (AKS). AKS is good for high-scale production deployments and provides autoscaling, and fast response times. [22_deployment_on_azure_kubernetes_service](22_deployment_on_azure_kubernetes_service.ipynb) explains how to deploy a web service to AKS through Azure ML.


