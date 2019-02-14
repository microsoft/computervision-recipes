# Image Classification
A large number of problems in the computer vision domain can be solved using image classification. These problems include building models to answer questions such as:
- Is an object present in the image? For example "dog", "cat", "ship", etc...
- What kind of geographic land (forest? desert? river?) is in this satelite image?

In these notebooks, we learn how to use fast.ai and Azure to train, test, and deploy an image classification model. We will start with multi-class classification and then go into multi-label classification. 

To build and deploy an image classification model using fast.ai and Azure, you go through the following steps:

__Model development__
1. Preparing the data (getting images and labels into the correct format)
1. Create data loaders
1. Explore your data
1. Data augmentation
1. Building and iterating on the model (architecture selection, hyperparameter tuning, evaluating results, etc)

__Model deployment__
1. Deploying the model

## Prerequisites

__Model development__

1. fastai>=1.0
1. GPU enabled machine

__Model deployment__

1. If you don't have an Azure subscription, create a free account before you begin. 
1. Set up an Azure Machine Learning workspace

## Notebooks

1. [Multiclass Classification - Classifying Dog Breeds](multiclass_classification.ipynb)
1. [Multilabel Classification - Classifying MET Lithographs](multilabel_classification.ipynb)
1. [Deploying our classification model](.)

## Appendix 

1. [Fastai course v3](https://github.com/fastai/course-v3)

