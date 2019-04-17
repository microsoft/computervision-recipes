# Computer Vision

This repository provides implementations and best practice guidelines for building Computer Vision systems. All examples are given as Jupyter notebooks, and use PyTorch as Deep Learning library.

[![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/Build-UnitTest?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=2&branchName=staging)

## Overview

The goal of this repository is to help speed up development of Computer Vision applications. Rather than implementing custom approaches, the focus is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out often observed pitfalls, and show how to use the cloud for deployment and training.

Currently, the main investment/priority is around image classification and to a lesser extend image segmentation. We also actively work on providing a basic (but often sufficiently accurate) example on how to do image similarity. Object detection is scheduled to start once image classification is completed. See the projects and milestones in this repository for more details.


## Getting Started

Instructions on how to get started, as well as our example notebooks and discussions are provided in the [image classification](image_classification/README.md) subfolder.

Note that for certain Computer Vision problems, ready-made or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if any of these address the problem at hand. Only if that is not the case, or if the accuracy of these solutions is not sufficient, do we recommend the much more time-consuming and difficult (since it requires expert knowledge) path of building custom models.

These Microsoft  services address common Computer Vision tasks:

- [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/)
Pre-trained REST APIs which can be called to do e.g. image classification, face recognition, OCR, video analytics, and much more. These APIs are easy to use and work out of the box (e.g. no training required), however customization is limited. See the various demos to get a feeling for their functionality, e.g. on this [site](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/).


- [Custom Vision Service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
SaaS service to train and deploy a model as a REST API given a user-provided training set. All steps from image upload, annotation, to model deployment can either be performed using a UI, or alternatively (but not necessary) a Python SDK. Both training image classification and object detection models is supported, with only minimal machine learning knowledge. The Custom Vision Service hence offers more flexibility than using the pre-trained Cognitive Services APIs, but requires the user to bring and annotate their own datasets.

- [Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
Scenario-agnostic machine learning service that helps users accelerate training and deploying machine learning models. While not specific for Computer Vision workloads, one can use the AzureML Python SDK to deploy scalable and reliable web-services using e.g. Kubernetes, or for heavily parallel training on a cloud-based GPU cluster. While AzureML offers significantly more flexibility than the other options above, it also requires significantly more machine learning and programming knowledge.


## Computer Vision Domains

Most applications in Computer Vision fall into one of these 4 categories:

- **Image classification**: Given an input image, predict what objects are present. This is typically the easiest CV problem to solve, however requires objects to be reasonably large in the image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_ic_vis.jpg" height="150" alt="Image classification visualization"/>  

- **Object Detection**: Given an input image, predict what objects are present and where the objects are (using rectangular coordinates). Object detection approaches work even if the object is small. However model training takes longer than image classification, and manually annotating images is more time-consuming as both labels and rectangular coordinates must be provided.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_od_vis.jpg" height="150" alt="Object detect visualization"/>

- **Image Similarity** Given an input image, find all similar images in a reference dataset. Here, rather than predicting a label or a rectangle, the task is to sort a reference dataset by their similarity to the query image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_is_vis.jpg" height="150" alt="Image similarity visualization"/>

- **Image Segmentation** Given an input image, assign a label to all pixels e.g. background, bottle, hand, sky, etc. In practice, this problem is less common in industry, in big parts due to the (time-consuming to annotate) ground truth segmentation required during training.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img align="center" src="https://cvbp.blob.core.windows.net/public/images/document_images/intro_iseg_vis.jpg" height="150" alt="Image segmentation visualization"/>


## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).
