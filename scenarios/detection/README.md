# Object Detection

This directory provides examples and best practices for building object detection systems. Our goal is to enable the users to bring their own datasets and train a high-accuracy model easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and extensive documentation of common pitfalls, best practices, etc.

Object Detection is one of the main problems in Computer Vision. Traditionally, this required expert knowledge to identify and implement so called “features” that highlight the position of objects in the image. Starting in 2012 with the famous AlexNet paper, Deep Neural Networks are used to automatically find these features. This lead to a huge improvement in the field for a large range of problems.

This repository uses [torchvision's](https://pytorch.org/docs/stable/torchvision/index.html) Faster R-CNN implementation which has been shown to work well on a wide variety of Computer Vision problems. See the [FAQ](FAQ.md) for an explanation of the underlying data science aspects.

```diff
+ (August 2019) This is work-in-progress and more functionality and documentation will be added continuously.
```


## Frequently asked questions

Answers to frequently asked questions such as "How does the technology work?" can be found in the [FAQ](FAQ.md) located in this folder. For generic questions such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see the [FAQ.md](../classification/FAQ.md) in the classification folder.


## Getting Started

To get started, simply follow the repository-wide installation instructions in this [readme](../README.md/#getting-started) to create a conda environment called _cv_. No other steps are required. To activate the _cv_ environment run:
  ```
  conda activate cv
  ```

AT THIS POINT, INSTALLATION ON WINDOWS MACHINES REQUIRES A FEW EXTRA STEPS. THESE ARE:
- BEFORE RUNNING *conda env create -f environment.yml*
   - REMOVE *pycocotools>=2.0* FROM environment.yaml file
- AFTER RUNNING *conda env create -f environment.yml* AND *conda activate cv* ALSO RUN
   - pip install Cython
   - pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI

We recommend running these samples on a machine with GPU, on either Windows or Linux. While a GPU is technically not required, training gets prohibitively slow even when using only a few dozens of images.


## Notebooks

We provide several notebooks to show how object detection algorithms can be designed and evaluated:

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./00_webcam.ipynb)| Quick-start notebook which demonstrates how to build an object detection system using a single image or webcam as input.
| [01_training_and_evaluation_introduction.ipynb](./01_training_and_evaluation_introduction.ipynb)| Notebook which explains the basic concepts around model training and evaluation.|
| [11_exploring_hyperparameters_on_azureml.ipynb](./11_exploring_hyperparameters_on_azureml.ipynb)| Performs highly parallel parameter sweeping using AzureML's HyperDrive. |


## Contribution guidelines

See the [contribution guidelines](../../CONTRIBUTING.md) in the root folder.
