# Image classification

This directory provides examples and best practices for building image classification systems. Our goal is enable the users to bring their own datasets and train a high-accuracy classifier easily and quickly. To this end, we provide example notebooks with pre-set default parameters shown to work well on a variety of datasets, and extensive documentation of common pitfalls, best practices, etc. In addition, we show how to use the Azure cloud to e.g. deploy models as a webserivce, or to speed up training on large datasets using the power of the cloud.


We recommend to use PyTorch as Deep Learning library due to its ease of use, simple debugging, and popularity in the data science community. For Computer Vision functionality, we rely heavily on [fast.ai](https://github.com/fastai/fastai), one of the most well-known PyTorch data science libraries, which comes with rich feature support and extensive documentation. To get a better understanding of the underlying technology, we highly recommend to watch the [2019 fast.ai lecture series](https://course.fast.ai/videos/?lesson=1), and to go through fast.ai's [documentation](https://docs.fast.ai/).


## Notebooks

We provide several notebooks to show how image classification algorithms can be designed, evaluated and operationalized. Note that the notebooks starting with 0 are meant to be "required", while all other notebooks are optional.

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](.notebooks/00_webcam.ipynb)| Quick start notebooks which demonstrate how to load a trained model and run inference using a single image of webcam input.
| [01_training_introduction.ipynb](.notebooks/01_training_introduction.ipynb)| Notebook which explains some of the basic concepts around model training and evaluation.|
| [02_training_accuracy_vs_speed.ipynb](.notebooks/02_training_accuracy_vs_speed.ipynb)| Notebook to train a model with e.g. high accuracy of fast inference speed. <font color="orange"> Use this to train on your own datasets! </font> |
| [11_exploring_hyperparameters.ipynb](.notebooks/11_exploring_hyperparameters.ipynb)| Advanced notebook to find optimal parameters by doing an exhaustive grid search. |
| deployment/[01_deployment_on_azure_container_instances.ipynb](.notebooks/11_exploring_hyperparameters.ipynb)| Notebook showing how to deploy a trained model as REST API using Azure Container Instances. |

## Getting Started

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
1. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVisionBestPractices
    cd ComputerVisionBestPractices/image_classification
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
1. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
2. Run the [Webcam Image Classification Notebook](notebooks/00_webcam.ipynb) notebook under the notebooks folder. Make sure to change the kernel to "Python (cvbp)".



## Frequently asked questions

Expand each question below to see the answer. Note that some browsers do not render the drop-down correctly.

TODO:
- Might move to separate FAQ.md file
- Add more questions and initial text
- Check grammar and typos

<details>
<summary> How does the technology work? </summary>
State-of-the-art image classification methods such as used in this repository are based on Convolutional Neural Networks (CNN). CNNs are a special group of Deep Learning approaches shown to work extremely well on images. The key is to use CNNs which were already trained on millions of images (the ImageNet dataset) and to fine-tune these pre-trained CNNs using a potentially much smaller custom dataset. The web is full of sites explaining these conceptions, such as [link](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac).
</details>

<details>
<summary> Which problems can be solved using image classification, and which ones cannot? </summary>
Image classification can be used if the object-of-interest is relatively large in the image, e.g. not less than 10% image width/height. If the object is smaller, or if the location of the object is required, then object detection methods needs to be used instead.
</details>

<details>
<summary> How many images are required to train a model? </summary>
This depends heavily on the complexity of the images. For example, one needs less training images if the object-of-interest has always a similar visual appearance (same angle, same lighting conditions, etc), and more images otherwise. In practice, we have seen good results using 100 images for each class or sometime less. The only way to find out is to train a classifier with a few images, train a second time with more images, etc, and to observe how accuracy improves with more training images (while keeping the test set fixed).
</details>

<details>
<summary> How to annotate images? </summary>
Consistency is key. For example, either always annotate occluded objects, or never. Ambiguous image should be removed, eg if it is unclear to a human eye if an image shows a lemon or a tennis ball. Ensuring consistency is tricky especially if multiple people are involved, and hence our recommendation is that only a single person does the data annotation. Ideally the same person as the coder, since a lot can be learned by looking at the images.

Note that especially the test set should be of high annotation quality, to get reliable measure of accuracy.
</details>

<details>
<summary> How to split into training and test images? </summary>
Often a random split as is performed in the notebooks is fine. However, this is not always the case. For example, if the images were extracted from a movie, then having frame n in the training set and frame n+1 in the test set would result in accuracy estimates which are over-inflated since the training and test set might be too similar.
</details>

<details>
<summary> How to speed up training? </summary>
- Make sure images are stored on an SSD device, since otherwise HDD or network access times can dominate the training time.
- Very high-resolution images (>4 MegaPixels) should be downsized before DNN training since JPEG decoding is expensive and can slow down training by a factor of 10x.
</details>

<details>
<summary> How to improve accuracy or inference speed </summary>
See the [02_training_accuracy_vs_speed.ipynb](.notebooks/02_training_accuracy_vs_speed.ipynb) notebook for a discussion what parameters are important, and how to select a model which is fast during inference.
</details>





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
