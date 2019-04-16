# Image classification

This directory provides examples and best practices for building image classification systems. Our goal is enable the users to bring their own datasets and train a high-accuracy classifier easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and extensive documentation of common pitfalls, best practices, etc. In addition, we show how to use the Azure cloud to e.g. deploy models as a webserivce, or to speed up training on large datasets using the power of the cloud.


We recommend to use PyTorch as Deep Learning library due to its ease of use, simple debugging, and popularity in the data science community. For Computer Vision functionality, we rely heavily on [fast.ai](https://github.com/fastai/fastai), one of the most well-known PyTorch data science libraries, which comes with rich feature support and extensive documentation. To get a better understanding of the underlying technology, we highly recommend watching the [2019 fast.ai lecture series](https://course.fast.ai/videos/?lesson=1), and to go through fast.ai's [documentation](https://docs.fast.ai/).


## Frequently asked questions

Answers to Frequently Asked Questions such as "How many images do I need to train a model?" or "How to annotate images?" can be found in the [FAQ.md](FAQ.md) file.


## Notebooks

We provide several notebooks to show how image classification algorithms can be designed, evaluated and operationalized. Note that the notebooks starting with 0 are meant to be "required", while all other notebooks are optional.

Linux is the recommended operating system. While all notebooks can also be executed in Windows, there seems to be a bug in fast.ai which slows down training by an order of magnitude on small datasets. GPU is required for training, and for best training speeds we suggest using an Azure Data Science Virtual Machine with V100 GPU ([instructions](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm),[price table](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/)).


| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](notebooks/00_webcam.ipynb)| Quick start notebooks which demonstrate how to load a trained model and run inference using a single image of webcam input.
| [01_training_introduction.ipynb](notebooks/01_training_introduction.ipynb)| Notebook which explains some of the basic concepts around model training and evaluation.|
| [02_training_accuracy_vs_speed.ipynb](notebooks/02_training_accuracy_vs_speed.ipynb)| Notebook to train a model with e.g. high accuracy of fast inference speed. <font color="orange"> Use this to train on your own datasets! </font> |
| [10_image_annotation.ipynb](notebooks/10_image_annotation.ipynb)| Simple UI to annotate images. |
| [11_exploring_hyperparameters.ipynb](notebooks/11_exploring_hyperparameters.ipynb)| Advanced notebook to find optimal parameters by doing an exhaustive grid search. |
| [21_deployment_on_azure_container_instances.ipynb](notebooks/21_deployment_on_azure_container_instances.ipynb)| Notebook showing how to deploy a trained model as REST API using Azure Container Instances. |

## Getting Started

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
1. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVisionBestPractices
    cd ComputerVisionBestPractices/image_classification
    git checkout staging #for now we work in the staging directory
    ```
1. Install the conda environment
    ```
    conda env create -f environment.yml
    ```
1. Activate the conda environment and register it with Jupyter:
    ```
    conda activate cvbp
    python -m ipykernel install --user --name cvbp --display-name "Python (cvbp)"
    ```
    If you use *jupyterlab*, install jupyter-webrtc widget:
    ```
    jupyter labextension install jupyter-webrtc
    ``` 
1. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
1. Run the [Webcam Image Classification Notebook](notebooks/00_webcam.ipynb) notebook under the notebooks folder. Make sure to change the kernel to "Python (cvbp)".


## Coding guidelines

Variable naming should be consistent, i.e. an image should always be called "im" and not "i", "img", "imag", "image", etc. Since we take a strong dependency on fast.ai, variable naming should follow the standards of fast.ai which are described in this [abbreviation guide](https://docs.fast.ai/dev/abbr.html). The one exception to this guide is that variable names should be as self-explanatory as possible. For example, the meaning of the variable "batch_size" is clear, compared to using "bs" to refer to batch size.

See also the more general [coding guidelines](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines) in the "Microsoft/Recommenders" github repo.

The main variables and abbreviations are given in the table below:

| Abbreviation | Description |
| ------------ | ----------- |
| im                     | Image
| fig                    | Figure
| pt                     | 2D point (column,row)
| rect                   | Rectangle (order: left, top, right, bottom)
| width,height (or w/h)  | Image dimensions
| scale                  | Image up/downscaling factor
| angle                  | Rotation angle in degree
| table                  | 2D row/column matrix implemented using a list of lists
| row,list1D             | Single row in a table, i.e. single 1D-list
| rowItem                | Single item in a row
| line,string            | Single string
| lines,strings          | List of strings
| list1D                 | List of items, not necessarily strings
| -s    | Multiple of something (plural) should be indicated by appending an "s" to an abbreviation.
