# Human Action Recognition with R(2+1)D Model

R(2+1)D model architecture was originally presented in "A Closer Look at Spatiotemporal Convolutions for Action Recognition (2017)" paper from Facebook AI group.

<img src="model_arch.jpg" width="300" height="300" />

This project provides utility scripts to fine-tune the model and examples notebooks as follows:

| Notebook | Description |
| --- | --- |
| [r2p1d_hmdb51_finetuning](r2p1d_hmdb51_finetuning.ipynb) | An example of fine-tuning on the HMDB-51 dataset |
| [r2p1d_webcam](r2p1d_webcam.ipynb) | A real-time inferencing example on Webcam stream |
| [video_transformation](video_transformation.ipynb) | Examples of video transformations | 

Specifically, we use the model pre-trained on 65 million social media videos (IG) presented in "[Large-scale weakly-supervised pre-training for video action recognition (2019)](https://arxiv.org/abs/1905.00561)" paper.

The official model weights from [Facebook AI](https://github.com/facebookresearch/vmz) are for caffe2,
and we use PyTorch-converted weights from [here](https://github.com/moabitcoin/ig65m-pytorch).


## Installation
1. Setup conda environment
`conda env create -f environment.yml`

1. Activate the environment
`conda activate r2p1d`

1. Install jupyter kernel
`python -m ipykernel install --user --name r2p1d`

### (Optional) Mixed-precision training
* To use mixed-precision training via [NVIDIA-apex](https://github.com/NVIDIA/apex),
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


### WebCam tunneling
To run the model on remote GPU VM while using a local machine for WebCam streaming,

1. Open a terminal window that supports `ssh` from the local machine (e.g. [git-bash for Windows](https://gitforwindows.org/)).

1. Run following commandPort forward as follows (assuming Jupyter notebook will be running on the port 8888 on the VM) 
`ssh your-vm-address -L 8888:localhost:8888` 

1. Clone this repository from the VM and install the conda environment and Jupyter kernel.

1. Start Jupyter notebook from the VM without starting a browser.
`jupyter notebook --no-browser` 
You can also set Jupyter configuration to not start browser by default.

1. Copy the notebook address showing at the terminal and paste it to the browser on the local machine to open it.