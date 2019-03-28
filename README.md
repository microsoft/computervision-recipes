# Computer Vision Best Practices

This repository provides implementations and best practice guidelines for building Computer Vision systems. All examples are given as Jupyter notebooks, and use PyTorch as Deep Learning library.

[![Build Status](https://dev.azure.com/best-practices/computervision/_apis/build/status/Build-UnitTest?branchName=staging)](https://dev.azure.com/best-practices/computervision/_build/latest?definitionId=2&branchName=staging)

## Overview

The goal of this repository is to help speed up development of Computer Vision systems. Rather than implementing our own approaches, the focus of this repo is on providing examples and links to existing state-of-the-art libraries. In addition, having worked in this space for many years, we aim to answer common questions, point out often observed pitfalls, and show how to extend from local to cloud deployment and training.

Currently, the main investment/priority is around image classification and to a lesser extend image segmentation. We also actively work on providing a basic (but often sufficiently accurate) example on how to do image similarity. Object detection is currently scheduled to start once image classification is completed. See the projects and milestones in this repository for more details.


## Getting Started

Instructions on how to get started, as well as all example notebooks and discussions, are provided in the subfolder for [image classification](image_classification/README.md).

Note that for certain Computer Vision problems, ready-made or easily customizable solutions exist which do not require any custom coding. We strongly recommend evaluating if any of these address the problem at hand. Only if that is not the case, or if the accuracy of these solutions are not sufficient, do we recommend to go down the much more time-consuming and difficult (since it requires expert knowledge) path of building custom code.

These Microsoft  services address common Computer Vision tasks:

- [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/vision/): Pre-trained AI models which can be readily consumed as web-services to do e.g. image classification, face detection and recognition, OCR, and much more. See the various demos to get a feeling for what these APIs can do, e.g. [here](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/).



 allow you to consume
machine learning hosted services. Within Cognitive Services API, there are several
[computer vision services](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/):

   - [Face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/):
Face detection, person identification and emotion recognition
   - [Content Moderator](https://azure.microsoft.com/en-us/services/cognitive-services/content-moderator/):
Image, text and video moderation
   - [Computer Vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/):
 Analyzing images, reading text and handwriting, identifying celebrities, and intelligently generating thumbnails
   - [Video Indexer](https://azure.microsoft.com/en-us/services/media-services/video-indexer/):
 Analyzing videos

 Targeting popular and specific use cases, these services can be consumed with easy to use APIs.
Users do no have to do any modeling or understand any machine learning concepts. They simply need to pass an image
or video to the hosted endpoint, and consume the results that are returned.

 Note, for these Cognitive Services, the models are pretrained and cannot be modified.

- Custom Vision Service
[Custom Vision Service](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)
is a SaaS service where you can train your own vision models with minimal machine learning knowledge.
Upload labelled training images through the browser application or through their APIs and the Custom Vision Service
will help you train and evaluate your model. Once you are satisfied with your model's performance, the model will be
ready for consumption as an endpoint.

 Currently, the Custom Vision Service can do image classification (multi-class + multi-label) and object detection scenarios.

- Azure Machine Learning Service
[Azure Machine Learning service (AzureML)](https://azure.microsoft.com/en-us/services/machine-learning-service/)
is a scenario-agnostic machine learning service that will help users accelerate training and deploying
machine learning models. Use automated machine learning to identify suitable algorithms and tune hyperparameters faster.
Improve productivity and reduce costs with autoscaling compute and DevOps for machine learning.
Seamlessly deploy to the cloud and the edge with one click. Access all these capabilities from your favorite
Python environment using the latest open-source frameworks, such as PyTorch, TensorFlow, and scikit-learn.

---

## What Should I Use?
One approach is see if the scenario you are solving for is one that is covered by one of the Cognitive Services APIs.
If so, you can start by using those APIs and determine if the results are performant enough. If they are not,
you may consider customizing the model with the Custom Vision Service, or building your own model using
Azure Machine Learning service.

Another approach is to determine the degree of customizability and fine tuning you want.
Cognitive Services APIs provide no flexibility. The Custom Vision Service provides flexibility insofar as being able to
choose what kind of training data to use (it is also only limited so solving classification and object detection problems).
Azure Machine Learning service provides complete flexibility, letting you set hyperparameters, select model architectures
(or build your own), and perform any manipulation needed at the framework (pytorch, tensorflow, cntk, etc) level.

One consideration is that more customizability also translates to more responsibility.
When using Azure Machine Learning service, you get the most flexibility, but you will be responsible for making sure
the models are performant and deploying them.


## Computer Vision Domains

Most applications in Computer Vision fall into one of these 4 categories:

- **Image classification**: Given an input image, predict what objects are present. This is typically the easiest CV problem to solve, however require objects to be reasonably large in the image.
<img align="center" src="docs/media/intro_ic_vis.jpg" height="150" alt="Image classification visualization"/>  

- **Object Detection**: Given an input image, predict what objects are present and where the objects are (using rectangular co-ordinates). Object detection approaches work even if the object is small. However model training takes longer than image classification, and manually annotating images is more time-consuming (for more information see the annotation instructions for object detection LINK).
<img align="center" src="docs/media/intro_od_vis.jpg" height="150" alt="Object detect visualization"/>

- **Image Similarity** Given an input "query" image, find all similar images in a reference dataset. Here, rather than predicting a label or a rectangle, the task is to rank a provided reference dataset by their similarity to the query image.
<img align="center" src="docs/media/intro_is_vis.jpg" height="150" alt="Image similarity visualization"/>

- **Image Segmentation** Given an input "query" image, assign a label to all pixels if the pixel is background or one of the objects of interest. In practice, this problem is less common in industry, in big parts due to the exact segmentation masks required to train a model.
<img align="center" src="docs/media/intro_iseg_vis.jpg" height="150" alt="Image segmentation visualization"/>


## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).
