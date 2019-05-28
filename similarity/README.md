# Image similarity

This directory provides examples and best practices for building image similarity systems. Our goal is to enable the users to bring their own datasets and train a high-accuracy model easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and extensive documentation of common pitfalls, best practices, etc.

The majority of state-of-the-art systems for image similarity use DNNs to compute a representation of an image (e.g. a vector of 512 floating point values), and define similarity between two images by measuring the L2 distance, for instance, between the respective DNN representations.

The main difference between modern image similarity approaches is how the DNN is trained. A simple but quite powerful approach is to use a standard image classification loss - this is the approach taken in this repository, and explained in the [classification](../classification/README.md) folder. More accurate similarity measures are based on DNNs which are trained explicitly for image similarity, such as the [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) work which uses a Siamese network architecture. FaceNet-like approaches will be added to this repository at a later point.


## Notebooks

We provide several notebooks to show how image similarity algorithms can be designed and evaluated.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./notebooks/00_webcam.ipynb)| Quick start notebook which demonstrates how to build an image retrieval system using a single image or webcam as input.
| [01_training_and_evaluation_introduction.ipynb](./notebooks/01_training_and_evaluation_introduction.ipynb)| Notebook which explains the basic concepts around model training and evaluation, based on using DNNs trained for image classification.|

## Getting Started

To setup on your local machine follow the [Getting Started](../classification/#getting-started) section in the image classification folder. Furthermore, basic image classification knowledge explained by the notebooks [01_training_introduction.ipynb](../classification/notebooks/01_training_introduction.ipynb) and [02_training_accuracy_vs_speed.ipynb](../classification/notebooks/02_training_accuracy_vs_speed.ipynb) is assumed.


## Coding guidelines

See the [coding guidelines](../classification/#coding-guidelines) in the image classification folder.

## Frequently asked questions

Answers to Frequently Asked Questions such as "How many images do I need to train a model?" or "How to annotate images?" can be found in the [FAQ.md](FAQ.md) file. For image classification specified questions, see the [FAQ.md](../classification/FAQ.md) in the classification folder.
