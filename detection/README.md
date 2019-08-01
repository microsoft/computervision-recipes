# Object Detection

This directory provides examples and best practices for building object detection systems. Our goal is to enable the users to bring their own datasets and train a high-accuracy model easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and extensive documentation of common pitfalls, best practices, etc.

Object Detection is one of the main problems in Computer Vision. Traditionally, this required expert knowledge to identify and implement so called “features” that highlight the position of objects in the image. Starting in 2012 with the famous AlexNet paper, Deep Neural Networks are used to automatically find these features. This lead to a huge improvement in the field for a large range of problems.

This repository uses [torchvision's](https://pytorch.org/docs/stable/torchvision/index.html) Faster R-CNN implementation which has been shown to work well on a wide variety of Computer Vision problems. See the [FAQ](FAQ.md) for an introduction how the technology works.

<p> <span style="color:green"> (Aug 2019) This is work-in-progress and expect more functionality and documentation to be continuously added. </span> </p>


## Frequently asked questions

Answers to Frequently Asked Questions such as "How does the technology work?" or "Which problems can be solved using object detection?" can be found in the [FAQ.md](FAQ.md) file. For generic questions such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see the [FAQ.md](../classification/FAQ.md) in the classification folder.


## Getting Started

To get started, follow the repository-wide installation instructions in the root [readme](https://github.com/microsoft/ComputerVision/tree/staging#getting-started). This will create a conda environment called _cv_. In addition, the library torchvision needs to get installed by executing:
  ```
  conda activate cv
  pip install torchvision
  ```

We recommend running these samples on a machine with GPU, either Windows or Linux. While a GPU is technically not required, training can become very slow even on only 10s of images.


## Notebooks

We provide several notebooks to show how object detection algorithms can be designed and evaluated.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./notebooks/00_webcam.ipynb)| Quick start notebook which demonstrates how to build an object detection system using a single image or webcam as input.
| [01_training_and_evaluation_introduction.ipynb](./notebooks/01_training_and_evaluation_introduction.ipynb)| Notebook which explains the basic concepts around model training and evaluation.|

## Coding guidelines

See the [coding guidelines](../classification/#coding-guidelines) in the image classification folder.
