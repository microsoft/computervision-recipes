# Computer Vision
In recent years, we see an extra-ordinary growth in Computer Vision, with applications in face recognition, image understanding, search, drones, mapping, semi-autonomous and autonomous vehicles. Key essence to many of these applications are visual recognition tasks such  as image classification, object detection and image similarity. Researchers have been applying newer deep learning methods to achieve state-of-the-art(SOTA) results on these challenging visual recognition tasks. 

This repository provides examples and best practice guidelines for building computer vision systems. The focus of the repository is on state-of-the-art methods that are popular among researchers and practitioners working on problems involving image recognition, object detection and image similarity.

These examples are provided as [Jupyter notebooks](scenarios) and common [utility functions](utils_cv). All examples use PyTorch as the underlying deep learning library.

## Overview

The goal of this repository is to accelerate the development of computer vision applications. Rather than creating implementions from scratch, the focus is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out frequently observed pitfalls, and show how to use the cloud for training and deployment.

We hope that these examples and utilities can significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.

## Get Started

To get started, navigate to the [Setup Guide](SETUP.md), which lists
instructions on how to setup the compute environment and dependencies needed to run the
notebooks in this repo. Once your environment is setup, navigate to the
[Scenarios](scenarios) folder and start exploring the notebooks.

## Scenarios

The following is a summary of commonly used Computer Vision scenarios that are covered in this repository. For each of these scenarios, we give you the tools to effectively build your own model. This includes simple tasks such as fine-tuning your own model on your own data, to more complex tasks such as hard-negative mining and even model deployment. See all supported scenarios [here](scenarios).

| Scenario | Description |
| -------- | ----------- |
| [Classification](scenarios/classification) | Image Classification is a supervised machine learning technique that allows you to learn and predict the category of a given image. |
| [Similarity](scenarios/similarity)  | Image Similarity is a way to compute a similarity score given a pair of images. Given an image, it allows you to identify the most similar image in a given dataset.  |
| [Detection](scenarios/detection) | Object Detection is a supervised machine learning technique that allows you to detect the bounding box of an object within an image. |

## Computer Vision on Azure

Note that for certain computer vision problems, you may not need to build your own models. Instead, pre-built or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if these can sufficiently solve your problem. If these solutions are not applicable, or the accuracy of these solutions is not sufficient, then resorting to more complex and time-consuming custom approaches may be necessary.

The following Microsoft services offer simple solutions to address common computer vision tasks:

- [Vision Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/)
are a set of pre-trained REST APIs which can be called for image tagging, face recognition, OCR, video analytics, and more. These APIs work out of the box and require minimal expertise in machine learning, but have limited customization capabilities. See the various demos available to get a feel for the functionality (e.g. [Computer Vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#analyze)).

- [Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
is a SaaS service to train and deploy a model as a REST API given a user-provided training set. All steps including image upload, annotation, and model deployment can be performed using either the UI or a Python SDK. Training image classification or object detection models can be achieved with minimal machine learning expertise. The Custom Vision offers more flexibility than using the pre-trained cognitive services APIs, but requires the user to bring and annotate their own data.

If you need to train your own model, the following services and links provide additional information that is likely useful.

- [Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
is a service that helps users accelerate the training and deploying of machine learning models. While not specific for computer vision workloads, the AzureML Python SDK can be used for scalable and reliable training and deployment of machine learning solutions to the cloud. We leverage Azure Machine Learning in several of the notebooks within this repository (e.g. [deployment to Azure Kubernetes Service](classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb))

- [Azure AI Reference architectures](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-python-models) 
provide a set of examples (backed by code) of how to build common AI-oriented workloads that leverage multiple cloud components. While not computer vision specific, these reference architectures cover several machine learning workloads such as model deployment or batch scoring.

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



