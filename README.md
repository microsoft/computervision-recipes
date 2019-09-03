# Computer Vision

This repository provides examples and best practice guidelines for building computer vision systems. All examples are given as Jupyter notebooks, and use PyTorch as the deep learning library.

## Overview

The goal of this repository is to accelerate the development of computer vision applications. Rather than creating implementions from scratch, the focus is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out frequently observed pitfalls, and show how to use the cloud for training and deployment.

The current main priority is to support image classification. Additionally, we also provide a basic (but often sufficiently accurate) example for image similarity. See the [projects](https://github.com/Microsoft/ComputerVision/projects) and [milestones](https://github.com/Microsoft/ComputerVision/milestones) pages in this repository for more details.

## Getting Started

To get started:

1. (Optional) Create an Azure Data Science Virtual Machine with e.g. a V100 GPU ([instructions](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm), [price table](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/)). 
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html). This step can be skipped if working on a Data Science Virtual Machine.
1. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVision
    ```
1. Install the conda environment, you'll find the `environment.yml` file in the root directory. To build the conda environment:
    ```
    conda env create -f environment.yml
    ```
1. Activate the conda environment and register it with Jupyter:
    ```
    conda activate cv
    python -m ipykernel install --user --name cv --display-name "Python (cv)"
    ```
    If you would like to use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/), install `jupyter-webrtc` widget:
    ```
    jupyter labextension install jupyter-webrtc
    ```
1. Start the Jupyter notebook server
    ```
    jupyter notebook
    ```
1. At this point, you should be able to run the notebooks in this repo. Explore our notebooks on the following computer vision domains. Make sure to change the kernel to "Python (cv)".
    - [/classification](classification#notebooks)
    - [/similarity](similarity#notebooks)
    - [/detection](detection#notebooks)
    

As an alternative to the steps above, and if one wants to install only
the 'utils_cv' library (without creating a new conda environment),
this can be done by running

```bash
pip install git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv
```

or by downloading the repo and then running `pip install .` in the
root directory.

## Introduction

Note that for certain computer vision problems, you may not need to build your own models. Instead, pre-built or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if these can sufficiently solve your problem. If these solutions are not applicable, or the accuracy of these solutions is not sufficient, then resorting to more complex and time-consuming custom approaches may be necessary.

The following Microsoft services offer simple solutions to address common computer vision tasks:

- [Vision Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/)
are a set of pre-trained REST APIs which can be called for image tagging, face recognition, OCR, video analytics, and more. These APIs work out of the box and require minimal expertise in machine learning, but have limited customization capabilities. See the various demos available to get a feel for the functionality (e.g. [Computer Vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#analyze)).

- [Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
is a SaaS service to train and deploy a model as a REST API given a user-provided training set. All steps including image upload, annotation, and model deployment can be performed using either the UI or a Python SDK. Training image classification or object detection models can be achieved with minimal machine learning expertise. The Custom Vision offers more flexibility than using the pre-trained cognitive services APIs, but requires the user to bring and annotate their own data.

## Build Your Own Computer Vision Model

If you need to train your own model, the following services and links provide additional information that is likely useful.

- [Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
is a service that helps users accelerate the training and deploying of machine learning models. While not specific for computer vision workloads, the AzureML Python SDK can be used for scalable and reliable training and deployment of machine learning solutions to the cloud. We leverage Azure Machine Learning in several of the notebooks within this repository (e.g. [deployment to Azure Kubernetes Service](classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb))

- [Azure AI Reference architectures](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-python-models) 
provide a set of examples (backed by code) of how to build common AI-oriented workloads that leverage multiple cloud components. While not computer vision specific, these reference architectures cover several machine learning workloads such as model deployment or batch scoring.


## Computer Vision Domains

Most applications in computer vision (CV) fall into one of these 4 categories:

- **Image classification**: Given an input image, predict what object is present in the image. This is typically the easiest CV problem to solve, however classification requires objects to be reasonably large in the image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_ic_vis.jpg" height="150" alt="Image classification visualization"/>  

- **Object Detection**: Given an input image, identify and locate which objects are present (using rectangular coordinates). Object detection can find small objects in an image. Compared to image classification, both model training and manually annotating images is more time-consuming in object detection, since both the label and location are required.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_od_vis.jpg" height="150" alt="Object detect visualization"/>

- **Image Similarity** Given an input image, find all similar objects in images from a reference dataset. Here, rather than predicting a label and/or rectangle, the task is to sort through a reference dataset to find objects similar to that found in the query image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_is_vis.jpg" height="150" alt="Image similarity visualization"/>

- **Image Segmentation** Given an input image, assign a label to every pixel (e.g., background, bottle, hand, sky, etc.). In practice, this problem is less common in industry, in large part due to time required to label the ground truth segmentation required in order to train a solution.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_iseg_vis.jpg" height="150" alt="Image segmentation visualization"/>

## Build Status

### VM Testing

| Build Type | Branch | Status |  | Branch | Status |
| --- | --- | --- | --- | --- | --- |
| **Linux GPU** |  master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=13&branchName=master)  | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=13&branchName=staging) |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=18&branchName=master)| | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=18&branchName=staging)|
| **Windows GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=16&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=16&branchName=staging)|
| **Windows CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=17&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=17&branchName=staging)|
| **AzureML Notebooks** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml-notebook-test-linux-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=43&branchName=master)| | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml-notebook-test-linux-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=43&branchName=staging)|

### AzureML Testing

| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linxu GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/bp-azureml-unit-test-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=41&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/bp-azureml-unit-test-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=41&branchName=staging)|
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/aml-unit-test-linux-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=37&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/aml-unit-test-linux-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=37&branchName=staging)|
| **Notebook unit GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/azureml-unit-test-linux-nb-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=42&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/azureml-unit-test-linux-nb-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=42&branchName=staging) |
| **Nightly GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/nightly-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=46&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/nightly-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=46&branchName=staging) |


## Contributing
This project welcomes contributions and suggestions. Please see our [contribution guidelines](CONTRIBUTING.md).

## Data/Telemetry
The Azure Machine Learning image classification notebooks ([20_azure_workspace_setup](classification/notebooks/20_azure_workspace_setup.ipynb), [21_deployment_on_azure_container_instances](classification/notebooks/21_deployment_on_azure_container_instances.ipynb), [22_deployment_on_azure_kubernetes_service](classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb), [23_aci_aks_web_service_testing](classification/notebooks/23_aci_aks_web_service_testing.ipynb), and [24_exploring_hyperparameters_on_azureml](classification/notebooks/24_exploring_hyperparameters_on_azureml.ipynb)) collect browser usage data and send it to Microsoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement).

To opt out of tracking, please go to the raw `.ipynb` files and remove the following line of code (the URL will be slightly different depending on the file):

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/ComputerVision/classification/notebooks/21_deployment_on_azure_container_instances.png)"
```
