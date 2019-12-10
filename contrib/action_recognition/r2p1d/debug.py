import os
import time
import sys

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.cuda as cuda
import torch.nn as nn
import torchvision

from vu.data import show_batch, VideoDataset
from vu.models.r2plus1d import R2Plus1D 
from vu.utils import system_info

system_info()

DATA_ROOT = os.path.join("data", "custom")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
# This split is known as "split1"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train.txt")
TEST_SPLIT = os.path.join(DATA_ROOT, "val.txt")

# 8-frame or 32-frame models
MODEL_INPUT_SIZE = 32
# 16 for 8-frame model.
BATCH_SIZE = 8

# Model configuration
r2plus1d_custom_cfgs = dict(
    # HMDB51 dataset spec
    num_classes=2,
    video_dir=VIDEO_DIR,
    train_split=TRAIN_SPLIT,
    valid_split=TEST_SPLIT,
    # Pre-trained model spec ("Closer look" and "Large-scale" papers)
    base_model='ig65m',
    sample_length=MODEL_INPUT_SIZE,     
    sample_step=1,        # Frame sampling step
    im_scale=128,         # After scaling, the frames will be cropped to (112 x 112)
    mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989),
    random_shift=True,
    temporal_jitter_step=2,    # Temporal jitter step in frames (only for training set)
    flip_ratio=0.5,
    random_crop=True,
    video_ext='mp4',
)

# Training configuration
train_cfgs = dict(
    mixed_prec=False,
    batch_size=BATCH_SIZE,
    grad_steps=2,
    lr=0.001,         # 0.001 ("Closer look" paper, HMDB51)
    momentum=0.95,
    warmup_pct=0.3,  # First 30% of the steps will be used for warming-up
    lr_decay_factor=0.001,
    weight_decay=0.0001,
    epochs=48,
    model_name='custom',
    model_dir=os.path.join("checkpoints", "ig65m_kinetics"),
)

learn = R2Plus1D(r2plus1d_custom_cfgs)

learn.fit(train_cfgs)