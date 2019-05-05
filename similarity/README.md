# Image similarity

THIS README AND THE NOTEBOOKS ARE WORK IN PROGRESS. TODO:
- Documentation currently only in draft stage (edit readme, edit FAQ, use proof-read IC readme as template, etc).
- Notebook 00_ is currently only a rough draft.

This directory provides examples and best practices for building image similarity systems. Our goal is to enable the users to bring their own datasets and train an accurate system easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and documentation of common pitfalls, best practices, etc.

The majority of state-of-the-art systems for image similarity use a DNN to compute a representation of an image (e.g. a vector of 512 floating point number), and define similarity between two images as the e.g. L2 distance between the respective DNN representations. Such a similarity measure can be used in Image Retrieval systems where, given a *query* image, the goal is to find all similar images in a *reference* set. This can be implemented using the following steps:
1. Compute the DNN embeddings for all *reference* images and store on disk.
2. Compute the DNN embedding for the *query* image.
3. Evaluate the L2 distance between the query embedding and all reference embeddings.
4. Return the images with lowest distance or with distance lower than a specified threshold.

Many state-of-the-art approaches follow the approach outlined above and differ mainly how the DNN is trained. A simple but quite powerful approach is to train image classification DNNs - this is the approach taken in this repository. More accurate similarity measures are based on DNNs which are trained explicitly for image similarity, such as the [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) work which uses an Siamese network architecture. FaceNet-like approaches will be added to this repository at a later stage.


## Frequently asked questions

Answers to Frequently Asked Questions such as "How many images do I need to train a model?" or "How to annotate images?" can be found in the [FAQ.md](FAQ.md) file. For image classification specified questions, see the [FAQ.md](../classification/FAQ.md) in the classification folder.


## Notebooks

We provide several notebooks to show how image similarity algorithms can be designed and evaluated.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](../classification/notebooks/00_webcam.ipynb)| Quick start notebooks which demonstrate how to load a trained model and run image retrieval using a single image or webcam input.
| [01_classification_architecture.ipynb](../classification/notebooks/01_training_introduction.ipynb)| Notebook which explains some of the basic concepts around model training and evaluation, based on using DNN trained for image classification.|


## Getting Started

To setup on your local machine follow the [Getting Started](../classification/#getting-started) section in the image classification folder. Furthermore, basic image classification knowledge explained by the notebooks [01_training_introduction.ipynb](../classification/01_training_introduction.ipynb) and [02_training_accuracy_vs_speed.ipynb](../classification/02_training_accuracy_vs_speed.ipynb) is assumed.


## Coding guidelines

See the [coding guidelines](../classification/#coding-guidelines) in the image classification folder.
