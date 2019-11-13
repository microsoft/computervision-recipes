# Setup Guide

This document describes how to setup all the dependencies to run the notebooks
in this repository.

Many computer visions scenarios are extremely computationally heavy. Training a
model often requires a machine that has a GPU, and would otherwise be too slow.
We recommend using the GPU-enabled [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) since it comes prepared with a lot of the prerequisites needed to efficiently do computer vision.

To scale up or to operationalize your models, we recommend setting up [Azure
ML](https://docs.microsoft.com/en-us/azure/machine-learning/). Our notebooks
provide instructions on how to use it.


## Table of Contents

1. [Compute Environment](#compute-environments)
1. [System Requirements](#system-requirements)
1. [Installation](#installation)
1. [Tunneling](#tunneling)

## Compute Environments

Most computer vision scenarios require a GPU, especially if you're training a
custom model. We recommend using a virtual machine to run the notebooks on.
Specifically, we'll want one with a powerful GPU. The NVIDIA's Tesla V100 is a
good choice that can be found in most Azure regions.

The easiest way to get started is to use the [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/). This VM will come installed with all the system requirements that are needed to run the notebooks in this repository. If you choose this option, you can skip the [System Requirements](#system-requirements) step in this guide as those requirements come pre-installed on the DSVM.

Before creating your Azure DSVM, you need to decide what kind of VM Size you want. Some VMs have GPUs, some have multiple GPUs, and some don't have any GPUs at all. For this repo, we recommend selecting an Ubuntu VM of the size [Standard_NC6_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#ncv3-series). The Standard_NC6_v3 uses the Nvidia Tesla V100 which will help us train our computer vision models and iterate quickly. We also recommend using Linux (Ubuntu) over Windows as some of the dependencies in this repository run better on Linux. 

For users new to Azure, your subscription may not come with a quota for GPUs. You may need to go into the Azure portal to increase your quota for GPU vms. Learn more about how to do this here: https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits.

Here are some ways you can create the DSVM:

__Provision a Data Science VM with the Azure Portal or CLI__

You can also spin up a Data Science VM directly using the Azure portal. To do so, follow
[this](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)
link that shows you how to provision your Data Science VM through the portal.

You can alternatively use the Azure command line (CLI) as well. Follow
[this](https://docs.microsoft.com/en-us/cli/azure/azure-cli-vm-tutorial?view=azure-cli-latest)
link to learn more about the Azure CLI and how it can be used to provision
resources.

__Virtual Machine Builder__

One easy way to create your DSVM is to use the [VM Builder](contrib/vm_builder) tool located inside of the 'contrib' folder in the root directory of the repo. Simply run `python contrib/vm_builder/vm_builder.py` at the root level of the repo and this tool will preconfigure your virtual machine with the appropriate settings for working with this repository. 

> NOTE: the VM builder only works on Linux and Mac.

## System Requirement

__Requirements__

* A machine running Linux >= 16.04 LTS or Windows
* Miniconda or Anaconda with Python version >= 3.6.
    * This is pre-installed on Azure DSVM such that one can run the following steps directly. To setup on your local machine, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.
    * It is recommended to update conda to the latest version: `conda update -n base -c defaults conda`

> NOTE: For Image Classification, Windows is up to 10x slower in training than Linux. You can set `num_workers=0`, but even still it will be up to 2x slower.

> NOTE: For Object Detection, Windows is about 20% slower in training but about same speed for inference.

__Dependencies__

Make sure you have CUDA Toolkit version 9.0 or above installed on your machine. You can run the command below in your terminal to check.

```
nvcc --version
```

If you don't have CUDA Toolkit or don't have the right version, please download it from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Installation
To install the repo and its dependencies perform the following steps:

1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html). This step can be skipped if working on a Data Science Virtual Machine.
1. Clone the repository
    ```
    git clone https://github.com/Microsoft/ComputerVision
    ```
1. Install the conda environment, you'll find the `environment.yml` file in the root directory. To build the conda environment:
    ```
    conda env create -f environment.yml
    ```
1. Activate the conda environment and register it with Jupyter:
    ```
    conda activate cv
    python -m ipykernel install --user --name cv --display-name "Python (cv)"
    ```
1. Start the Jupyter notebook server
    ```
    jupyter notebook
    ```
1. At this point, you should be able to run the [notebooks](#scenarios) in this repo. 

__pip install__

As an alternative to the steps above, and if you only want to install
the 'utils_cv' library (without creating a new conda environment),
this can be done using pip install:

```bash
pip install git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv
```

> NOTE: if you install this repo using this method, you will not have the notebooks loaded by default.

## Tunneling

If your compute environment is on a VM in the cloud, you can open a tunnel from your VM to your local machine using the following command:
```
$ssh -L local_port:remote_address:remote_port  <username>@<server-ip>
```

For example, if I want to run `jupyter notebook --port 8888` on my VM and I
wish to run the Jupyter notebooks on my local broswer on `localhost:9999`, I
would ssh into my VM using the following command:

```
$ssh -L 9999:localhost:8888 <username>@<server-ip>
```

This command will allow your local machine's port `9999` to access your remote
machine's port `8888`.
