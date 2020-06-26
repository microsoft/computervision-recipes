## Fine-tuning I3D model on HMDB-51

In this section we provide code for training a Two-Stream Inflated 3D ConvNet (I3D), introduced in \[[1](https://arxiv.org/pdf/1705.07750.pdf)\].  Our implementation uses the Pytorch models (and code) provided in [https://github.com/piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) - which have been pre-trained on the Kinetics Human Action Video dataset - and fine-tunes the models on the HMDB-51 action recognition dataset. The I3D model consists of two "streams" which are independently trained models. One stream takes the RGB image frames from videos as input and the other stream takes pre-computed optical flow as input. At test time, the outputs of each stream model are averaged to make the final prediction. The model results are as follows:

| Model | Paper top 1 accuracy (average over 3 splits) | Our models top 1 accuracy (split 1 only) |
| ------- | -------| ------- |
| RGB | 74.8 | 73.7 |
| Optical flow | 77.1 | 77.5 |
| Two-Stream | 80.7 | 81.2 |

## Download and pre-process HMDB-51 data

Download the HMDB-51 video database from [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Extract the videos with
```
mkdir rars && mkdir videos
unrar x hmdb51-org.rar rars/
for a in $(ls rars); do unrar x "rars/${a}" videos/; done;
```

 Use code provided in [https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks) to preprocess the raw videos into split videos into RGB frames and compute optical flow frames:
 ```
 git clone https://github.com/yjxiong/temporal-segment-networks
 cd temporal-segment-networks
 bash scripts/extract_optical_flow.sh /path/to/hmdb51/videos /path/to/rawframes/output
```
Edit the _C.DATASET.DIR option in [default.py](default.py) to point towards the rawframes input data directory.

## Setup environment
Setup environment

```
conda env create -f environment.yaml
conda activate i3d
```

## Download pretrained models
Download pretrained models

```
bash download_models.sh
```

## Fine-tune pretrained models on HMDB-51

Train RGB model
```
python train.py --cfg config/train_rgb.yaml
```

Train flow model
```
python train.py --cfg config/train_flow.yaml
```

Evaluate combined model
```
python test.py
```

\[1\] J. Carreira and A. Zisserman. Quo vadis, action recognition?
a new model and the kinetics dataset. In CVPR, 2017.
