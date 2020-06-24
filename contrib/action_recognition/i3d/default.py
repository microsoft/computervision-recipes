# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = "log"
_C.WORKERS = 16
_C.PIN_MEMORY = True
_C.SEED = 42

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True

# Dataset
_C.DATASET = CN()
_C.DATASET.SPLIT = 1
_C.DATASET.DIR = "/datadir/rawframes/"
_C.DATASET.NUM_CLASSES = 51

# NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "i3d_flow"
_C.MODEL.PRETRAINED_RGB = "pretrained_models/rgb_imagenet_kinetics.pt"
_C.MODEL.PRETRAINED_FLOW = "pretrained_models/flow_imagenet_kinetics.pt"
_C.MODEL.CHECKPOINT_DIR = "checkpoints"

# Train
_C.TRAIN = CN()
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.INPUT_SIZE = 224
_C.TRAIN.RESIZE_MIN = 256
_C.TRAIN.SAMPLE_FRAMES = 64
_C.TRAIN.MODALITY = "flow"
_C.TRAIN.BATCH_SIZE = 24
_C.TRAIN.GRAD_ACCUM_STEPS = 4
_C.TRAIN.MAX_EPOCHS = 50

# Test
_C.TEST = CN()
_C.TEST.EVAL_FREQ = 5
_C.TEST.PRINT_FREQ = 250
_C.TEST.BATCH_SIZE = 1
_C.TEST.MODALITY = "combined"
_C.TEST.MODEL_RGB = "pretrained_models/rgb_hmdb_split1.pt"
_C.TEST.MODEL_FLOW = "pretrained_models/flow_hmdb_split1.pt"

def update_config(cfg, options=None, config_file=None):
    cfg.defrost()

    if config_file:
        cfg.merge_from_file(config_file)

    if options:
        cfg.merge_from_list(options)

    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(_C, file=f)