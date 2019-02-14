

# Image classification

This directory provides examples and best practices for building image classification systems. We use [fast.ai](https://github.com/fastai/fastai) heavily, since it has rich feature support and ...


## Getting Started

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
2. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVisionBestPractices
    ```
3. [work in progress]
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
5. Run the [SAR Python CPU Movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) notebook under the 00_quick_start folder. Make sure to change the kernel to "Python (reco)".

## Coding guidelines

Variable naming should be consistent, i.e. an image should always be called "img" and not "i", "im", "img", "imag", "image", etc. Since for image classification we take on a heavy dependency on fast.ai, variable naming should follow the standards of fast.ai which are described in this [abbreviation guide](https://docs.fast.ai/dev/abbr.html).

The one exception to this guide is that variable names should be as self-explanatory as possible. For example, the meaning of the variable "batch_size" is clear, compared to using "bs" to refer to batch size.

The main variables and abbreviations are given in the table below:

| Abbreviation | Description |
| ------------ | ----------- |
| im                     | image
| fig                    | figure
| pt                     | 2D point (column,row)
| rect                   | rectangle (order: left, top, right, bottom)
| width,height (or w/h)  | image dimensions
| angle                  | rotation angle in degree
| scale                  | image up/downscaling factor
| table                  | 2D row/column matrix implemented using a list of lists
| row,list1D             | single row in a table, i.e. single 1D-list
| rowItem                | single item in a row
| line,string            | single string
| lines,strings          | list of strings
| list1D                 | list of items, not necessarily strings
| -s    | Multiple of something (plural) should be indicated by appending an "s" to an abbreviation.

## Notebooks

We provide several notebooks to show how image classification algorithms can be designed, evaluated and operationalized.

- [work in progress]
