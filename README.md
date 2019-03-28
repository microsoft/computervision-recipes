# Computer Vision Best Practices

This repository provides implementations and best practice guidelines for building Computer Vision systems. All examples are given as Jupyter notebooks, and use PyTorch as Deep Learning library.

[![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/Build-UnitTest?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=2&branchName=staging)

## Overview

The goal of this repository is to help speed up development of Computer Vision systems. Rather than implementing our own approaches, the focus of this repo is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out often observed pitfalls, and show how to extend from local to cloud deployment and training.

Currently, the main investment/priority is around image classification and to a lesser extend image segmentation. We also actively work on providing a basic (but often sufficiently accurate) example on how to do image similarity. Object detection is currently scheduled to start once image classification is completed. See the projects and milestones in this repository for more details.


## Getting Started

Instructions on how to get started, as well as all example notebooks and discussions, are provided in the subfolder for [image classification](image_classification/README.md).

Note that for certain Computer Vision problems, ready-made or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if any of these address the problem at hand. Only if that is not the case, or if the accuracy of these solutions turns it not sufficient, do we recommend the much more time-consuming and difficult (since it requires expert knowledge) path of building custom code.

These Microsoft  services address common Computer Vision tasks:

- [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/)
Pre-trained REST APIs which can be readily consumed to do e.g. image classification, face recognition, OCR, video analytics, and much more. See the various demos to get a feeling for what these APIs can do, e.g. [here](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/). These APIs are easy to use and work out of the box (e.g. no training required). However customization is either limited or not possible.


- [Custom Vision Service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
SaaS service to train a model and deploy a REST API given a user-specific training set. All steps from image upload, annotation, to model deployment can either be performed using a UI, or alternatively (but not necessary) a Python SDK. With the Custom Vision Service one can train image classification and object detection models with only minimal or no minimal machine learning knowledge. It hence offers more flexibility than using the pre-trained Cognitive Services APIs.

- [Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
is a scenario-agnostic machine learning service that will help users accelerate training and deploying machine learning models. While not specific for Computer Vision workloads, one can use AzureML to deploy scalable and reliable web-services using e.g. Kubernetes, or for heavily parallel training on a cloud-based GPU cluster. AzureML comes with Python SDK, which is the way we will use it. While AzureML offers significantly more flexibility than the other options above, it also requires significantly more machine learning and programming knowledge.


## Computer Vision Domains

Most applications in Computer Vision fall into one of these 4 categories:

- **Image classification**: Given an input image, predict what objects are present. This is typically the easiest CV problem to solve, however require objects to be reasonably large in the image.
<img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_ic_vis.jpg" height="150" alt="Image classification visualization"/>  

- **Object Detection**: Given an input image, predict what objects are present and where the objects are (using rectangular co-ordinates). Object detection approaches work even if the object is small. However model training takes longer than image classification, and manually annotating images is more time-consuming (for more information see the annotation instructions for object detection LINK).
<img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_od_vis.jpg" height="150" alt="Object detect visualization"/>

- **Image Similarity** Given an input "query" image, find all similar images in a reference dataset. Here, rather than predicting a label or a rectangle, the task is to rank a provided reference dataset by their similarity to the query image.
<img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_is_vis.jpg" height="150" alt="Image similarity visualization"/>

- **Image Segmentation** Given an input "query" image, assign a label to all pixels if the pixel is background or one of the objects of interest. In practice, this problem is less common in industry, in big parts due to the exact segmentation masks required to train a model.
<img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_iseg_vis.jpg" height="150" alt="Image segmentation visualization"/>


## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).
