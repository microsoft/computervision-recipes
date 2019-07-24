# Computer Vision

This repository provides implementations and best practice guidelines for building Computer Vision systems. All examples are given as Jupyter notebooks, and use PyTorch as Deep Learning library.

## Build Status

### VM Testing

| Build Type | Branch | Status |  | Branch | Status |
| --- | --- | --- | --- | --- | --- |
| **Linux GPU** |  master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=13&branchName=master)  | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=13&branchName=staging) |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=18&branchName=master)| | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-linux-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=18&branchName=staging)|
| **Windows GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=16&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=16&branchName=staging)|
| **Windows CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=17&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/unit-test-windows-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=17&branchName=staging)|

### AzureML Testing

| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linxu GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/bp-azureml-unit-test-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=41&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/bp-azureml-unit-test-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=41&branchName=staging)|
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/aml-unit-test-linux-cpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=37&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/aml-unit-test-linux-cpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=37&branchName=staging)|
| **Notebook unit GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/azureml-unit-test-linux-nb-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=42&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/azureml-unit-test-linux-nb-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=42&branchName=staging) |
| **Nightly GPU** | master | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/nightly-linux-gpu?branchName=master)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=46&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/azureml/nightly-linux-gpu?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=46&branchName=staging) |


## Overview

The goal of this repository is to help speed up development of Computer Vision applications. Rather than implementing custom approaches, the focus is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out often observed pitfalls, and show how to use the cloud for deployment and training.

Currently the main investment/priority is around image classification and to a lesser extend image segmentation. We are also actively working on providing a basic (but often sufficiently accurate) example for image similarity. Object detection is scheduled to start once image classification is completed. See the [projects](https://github.com/Microsoft/ComputerVision/projects) and [milestones](https://github.com/Microsoft/ComputerVision/milestones) pages in this repository for more details.

## Getting started

To get started on your local machine:

1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
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
    - /object detection [coming_soon]
    - /segmentation [coming_soon]

## Services

Note that for certain Computer Vision problems, you may not need to build your own models. Instead, ready-made or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if these can sufficiently solve your problem. If these solutions are not applicable, or the accuracy of these solutions is not sufficient, then resorting to more complex and time-consuming custom approaches may be necessary.

The following Microsoft services offer simple solutions to address common Computer Vision tasks:

- [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/)
provides pre-trained REST APIs which can be called for image classification, face recognition, OCR, video analytics, and much more. These APIs are easy to use and work out of the box (no training required), however customization is limited. See the various demos available for each domain to get a feel for the functionality (e.g., [computer vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/), [speech to text](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/) ).

- [Custom Vision Service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
is a SaaS service to train and deploy a model as a REST API given a user-provided training set. All steps from image upload, annotation, to model deployment can be performed using either the UI or a Python SDK. Training image classification or object detection models are supported using only minimal machine learning knowledge. The Custom Vision Service offers more flexibility than using the pre-trained Cognitive Services APIs, but requires the user to bring and annotate their own data.

- [Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
is a more general machine learning service that helps users accelerate training and deploying machine learning models. While not specific for Computer Vision workloads, the AzureML Python SDK can be used for scalable and reliable training and deploying machine learning solutions to the cloud. While AzureML offers significantly more flexibility than other options, it also requires significantly more machine learning and programming knowledge.

- [Azure AI Reference architectures](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-python-models) While not Computer Vision specific, these reference architectures cover general Machine Learning aspects such as model deployment or batch scoring.


## Computer Vision Domains

Most applications in Computer Vision (CV) fall into one of these 4 categories:

- **Image classification**: Given an input image, predict what object is present in the image. This is typically the easiest CV problem to solve, however classification requires objects to be reasonably large in the image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_ic_vis.jpg" height="150" alt="Image classification visualization"/>  

- **Object Detection**: Given an input image, identify and locate which objects are present (using rectangular coordinates). Object detection can find small objects in an image. Compared to image classification, both model training and manually annotating images is more time-consuming in object detection, since both the label and location are required.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_od_vis.jpg" height="150" alt="Object detect visualization"/>

- **Image Similarity** Given an input image, find all similar objects in images from a reference dataset. Here, rather than predicting a label and/or rectangle, the task is to sort through a reference dataset to find objects similar to that found in the query image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_is_vis.jpg" height="150" alt="Image similarity visualization"/>

- **Image Segmentation** Given an input image, assign a label to every pixel (e.g., background, bottle, hand, sky, etc.). In practice, this problem is less common in industry, in large part due to time required to label the ground truth segmentation required in order to train a solution.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="./media/intro_iseg_vis.jpg" height="150" alt="Image segmentation visualization"/>

## Contributing
This project welcomes contributions and suggestions. Please see our [contribution guidelines](CONTRIBUTING.md).

## Data/Telemetry
The Azure Machine Learning image classification notebooks ([20_azure_workspace_setup](classification/notebooks/20_azure_workspace_setup.ipynb), [21_deployment_on_azure_container_instances](classification/notebooks/21_deployment_on_azure_container_instances.ipynb), [22_deployment_on_azure_kubernetes_service](classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb), and [23_aci_aks_web_service_testing](classification/notebooks/23_aci_aks_web_service_testing.ipynb)) collect browser usage data and send it to Microsoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement).

To opt out of tracking, please go to the raw `.ipynb` files and remove the following line of code (the URL will be slightly different depending on the file):

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/ComputerVision/classification/notebooks/21_deployment_on_azure_container_instances.png)"
```

