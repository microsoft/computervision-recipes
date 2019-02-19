# Image classification

This directory provides examples and best practices for building image classification systems. We use [fast.ai](https://github.com/fastai/fastai) heavily, since it has rich feature support and ... .

See also fast.ai's [documentation](https://docs.fast.ai/) and most recent [course](https://github.com/fastai/course-v3) for more explanations and code examples.

In this directory, we will cover the following:

__Model development__
1. Preparing the data (getting images and labels into the correct format)
1. Create data loaders
1. Explore your data
1. Data augmentation
1. Building and iterating on the model (architecture selection, hyperparameter tuning, evaluating results, etc)

__Model deployment__
1. Deploying the model

## Getting Started

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
2. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVisionBestPractices
    ```
3. Enter the image classification directory and install the conda environment
   ```
   cd ComputerVisionBestPractices/image_classification
   conda env create -f environment.yml
   ```
4. Activate the conda environment and register it with Jupyter:
    ```
    conda activate cvbp
    python -m ipykernel install --user --name cvbp --display-name "Python (cvbp)"
    ```
5. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
5. [TODO] Run the [SAR Python CPU Movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) notebook under the 00_quick_start folder. Make sure to change the kernel to "Python (reco)".

## Coding guidelines

Variable naming should be consistent, i.e. an image should always be called "im" and not "i", "img", "imag", "image", etc. Since we take a strong dependency on fast.ai, variable naming should follow the standards of fast.ai which are described in this [abbreviation guide](https://docs.fast.ai/dev/abbr.html).

The one exception to this guide is that variable names should be as self-explanatory as possible. For example, the meaning of the variable "batch_size" is clear, compared to using "bs" to refer to batch size.

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

## Notebooks

We provide several notebooks to show how image classification algorithms can be designed, evaluated and operationalized.

- [work in progress]

## Appendix 

1. [Fastai course v3](https://github.com/fastai/course-v3)


