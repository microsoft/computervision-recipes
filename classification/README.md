# Image classification

This directory provides examples and best practices for building image classification systems. Our goal is enable users to easily and quickly train high-accuracy classifiers on their own datasets. We provide example notebooks with pre-set default parameters that are shown to work well on a variety of data sets. We also include extensive documentation of common pitfalls and best practices. Additionally, we show how Azure, Microsoft's cloud computing platform, can be used to speed up training on large data sets or deploy models as web services.

We recommend using PyTorch as a Deep Learning platform for its ease of use, simplicity when debugging, and popularity in the data science community. For Computer Vision functionality, we also rely heavily on [fast.ai](https://github.com/fastai/fastai), a PyTorch data science library which comes with rich deep learning features and extensive documentation. We highly recommend watching the [2019 fast.ai lecture series](https://course.fast.ai/videos/?lesson=1) video to understand the underlying technology. Fast.ai's [documentation](https://docs.fast.ai/) is also a valuable resource.

## Frequently asked questions

Answers to Frequently Asked Questions such as "How many images do I need to train a model?" or "How to annotate images?" can be found in the [FAQ.md](FAQ.md) file.

## Notebooks

We provide several notebooks to show how image classification algorithms are designed, evaluated and operationalized. Notebooks starting with `0` are intended to be run sequentially, as there are dependencies between them. These notebooks contain introductory "required" material. Notebooks starting with `1` can be considered optional and contain more complex and specialized topics.

While all notebooks can be executed in Windows, we have found the computer vision tools are often faster and more stable on the Linux operating system. Additionally, using GPU dramatically improves training speeds. We suggest using an Azure Data Science Virtual Machine with V100 GPU ([instructions](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm), [price table](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/)). 

We have also found that some browsers do not render Jupyter widgets correctly. If you have issues, try using an alternative browser, such as Chrome.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](notebooks/00_webcam.ipynb)| Demonstrates a trained model and inference an image from your computer's webcam.
| [01_training_introduction.ipynb](notebooks/01_training_introduction.ipynb)| Introduces some of the basic concepts around model training and evaluation.|
| [02_training_accuracy_vs_speed.ipynb](notebooks/02_training_accuracy_vs_speed.ipynb)| Trains a model with high accuracy vs one with a fast inferencing speed. *<font color="orange"> Use this to train on your own datasets! </font>* |
| [10_image_annotation.ipynb](notebooks/10_image_annotation.ipynb)| A simple UI to annotate images. |
| [11_exploring_hyperparameters.ipynb](notebooks/11_exploring_hyperparameters.ipynb)| Finds optimal model parameters using grid search. |
| [21_deployment_on_azure_container_instances.ipynb](notebooks/21_deployment_on_azure_container_instances.ipynb)| Deploys a trained model as REST API using Azure Container Instances. |

## Getting Started

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
1. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVisionBestPractices
    cd ComputerVisionBestPractices/image_classification
    git checkout staging # for now we work in the staging directory
    ```
1. Install the conda environment, you'll find the `environment.yml` file in the `classification` subdirectory. From there:
    ```
    conda env create -f environment.yml
    ```
1. Activate the conda environment and register it with Jupyter:
    ```
    conda activate cvbp
    python -m ipykernel install --user --name cvbp --display-name "Python (cvbp)"
    ```
    If you would like to use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/), install `jupyter-webrtc` widget:
    ```
    jupyter labextension install jupyter-webrtc
    ``` 
1. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
1. Start with the [00_webcam](notebooks/00_webcam.ipynb) Image Classification Notebook under the `notebooks` folder. Make sure to change the kernel to "Python (cvbp)".

## Coding guidelines

Since we take a strong dependency on fast.ai, variable naming should follow the standards of fast.ai which are described in this [abbreviation guide](https://docs.fast.ai/dev/abbr.html). For example, in computer vision cases, an image should always be abbreviated with `im` and not `i`, `img`, `imag`, `image`, etc. The one exception to this guide is that variable names should be as self-explanatory as possible. For example, the meaning of the variable `batch_size` is clearer than `bs` to refer to batch size.

More general [coding guidelines](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines) are avialable through in the [Microsoft/Recommenders](https://github.com/Microsoft/Recommenders) github repo.

The main variables and abbreviations are given in the table below:

| Abbreviation | Description |
| ------------ | ----------- |
| `im `                    | Image
| `fig`                    | Figure
| `pt`                     | 2D point (column,row)
| `rect`                   | Rectangle (order: left, top, right, bottom)
| `width`, `height`, `w`, `h`  | Image dimensions
| `scale`                  | Image up/down scaling factor
| `angle`                  | Rotation angle in degree
| `table`                  | 2D row/column matrix implemented using a list of lists
| `row`, `list1D`             | Single row in a table, i.e. single 1D-list
| `rowItem`                | Single item in a row
| `line`, `string`            | Single string
| `lines`, `strings`          | List of strings
| `list1D`                 | List of items, not necessarily strings
| -`s`    | Multiple of something (plural) should be indicated by appending an `s` to an abbreviation.
