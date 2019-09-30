# Setup Guide

This document describes how to setup all the dependencies to run the notebooks
in this repository.

Many computer visions scenarios are extremely computationlly heavy. Training a
model often requires a machine that has a GPU, and would otherwise be too slow.
We recommend using the GPU-enabled [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) since it comes prepared with a lot of the prerequisites needed to efficiently do computer vision.

To scale up or to operationalize your models, we recommend setting up [Azure
ML](https://docs.microsoft.com/en-us/azure/machine-learning/). Our notebooks
provide instructions on how to use it.


## Table of Contents

1. [Compute Environment](#compute-environments)
1. [System Requirements](#system-requirements)
1. [Installation](#installation)

## Compute Environments

Most computer vision scenarios require a GPU, especially if you're training a
custom model. We recommend using a virtual machine to run the notebooks on.
Specifically, we'll want one with a powerful GPU. The NVIDIA's Tesla V100 is a
good choice that can be found in most Azure regions.

The easiest way to get started is to use the [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/). This VM will come installed with all the system requirements that are needed to run the notebooks in this repository. If you choose this option, you can skip the [System Requirements](#system-requirements) step in this guide as those requirements come pre-installed on the DSVM.

Here are some ways you can create the DSVM:

__Virtual Machine Builder__

One easy way to create your DSVM is to use the [VM Builder](../contrib/vm_builder) tool located inside of the 'contrib' folder in the root directory of the repo. Simply run `python contrib/vm_builder/vm_builder.py` at the root level of the repo and this tool will preconfigure your virtual machine with the appropriate settings for working with this repository.

__Provision a Data Science VM with the Azure Portal or CLI__

You can also spin up a Data Science VM directly using the Azure portal. To do so, follow
[this](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)
link that shows you how to provision your Data Science VM through the portal.

You can alternatively use the Azure command line (CLI) as well. Follow
[this](https://docs.microsoft.com/en-us/cli/azure/azure-cli-vm-tutorial?view=azure-cli-latest)
link to learn more about the Azure CLI and how it can be used to provision
resources.

Once your virtual machine has been created, ssh and tunnel into the machine, then run the "Getting started" steps inside of it. 

## System Requirement
<TODO>


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

__Tunneling Your Notebooks__

If your compute environment is on a VM in the cloud, you can open a tunnel from your VM to your local machine using the following command:
```
$ssh -L local_port:remote_address:remote_port  <username>@<server-ip>
```

For example, if I want to run `jupyter notebook --port 8888` on my VM and I
wish to run the Jupyter notebooks on my local broswer on `localhost:9999`, I
would ssh into my VM using the following commend:
```
$ssh -L 9999:localhost:8888 <username>@<server-ip>
```

This command will allow your local machine's port 9999 to access your remote
machine's port 8888.

__pip install__

As an alternative to the steps above, and if you only want to install
the 'utils_cv' library (without creating a new conda environment),
this can be done using pip install:

```bash
pip install git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv
```
