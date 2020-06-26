# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
import sys
import numpy as np

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from videotransforms import (
    GroupScale, GroupCenterCrop, GroupNormalize, Stack
)

from models.pytorch_i3d import InceptionI3d

from metrics import accuracy, AverageMeter

from dataset import I3DDataSet
from default import _C as config
from default import update_config

# to work with vscode debugger https://github.com/joblib/joblib/issues/864
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def load_model(modality, state_dict_file):

    channels = 3 if modality == "RGB" else 2
    model = InceptionI3d(config.DATASET.NUM_CLASSES, in_channels=channels)
    state_dict = torch.load(state_dict_file)
    model.load_state_dict(state_dict)
    model = model.cuda()

    return model


def test(model, test_loader, modality):

    model.eval()

    target_list = []
    predictions_list = []
    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(test_loader):
            target_list.append(target)
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=2)
            predictions_list.append(output)

            if step % config.TEST.PRINT_FREQ == 0:
                print(('Step: [{0}/{1}]'.format(step, len(test_loader))))
    
    targets = torch.cat(target_list)
    predictions = torch.cat(predictions_list)
    return targets, predictions


def run(*options, cfg=None):

    update_config(config, options=options, config_file=cfg)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Setup Augmentation/Transformation pipeline
    input_size = config.TRAIN.INPUT_SIZE
    resize_range_min = config.TRAIN.RESIZE_MIN

    # Data-parallel
    devices_lst = list(range(torch.cuda.device_count()))
    print("Devices {}".format(devices_lst))

    if (config.TEST.MODALITY == "RGB") or (config.TEST.MODALITY == "combined"):

        rgb_loader = torch.utils.data.DataLoader(
            I3DDataSet(
                data_root=config.DATASET.DIR,
                split=config.DATASET.SPLIT,
                modality="RGB",
                train_mode=False,
                sample_frames_at_test=False,
                transform=torchvision.transforms.Compose([
                    GroupScale(config.TRAIN.RESIZE_MIN),
                    GroupCenterCrop(config.TRAIN.INPUT_SIZE),
                    GroupNormalize(modality="RGB"),
                    Stack(),
                ])
            ),
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        rgb_model_file = config.TEST.MODEL_RGB
        if not os.path.exists(rgb_model_file):
            raise FileNotFoundError(rgb_model_file, " does not exist")
        rgb_model = load_model(modality="RGB", state_dict_file=rgb_model_file)

        print("scoring with rgb model")
        targets, rgb_predictions = test(rgb_model, rgb_loader, "RGB")

        del rgb_model
        
        targets = targets.cuda(non_blocking=True)
        rgb_top1_accuracy = accuracy(rgb_predictions, targets, topk=(1, ))
        print("rgb top1 accuracy: ", rgb_top1_accuracy[0].cpu().numpy().tolist())
    
    if (config.TEST.MODALITY == "flow") or (config.TEST.MODALITY == "combined"):

        flow_loader = torch.utils.data.DataLoader(
            I3DDataSet(
                data_root=config.DATASET.DIR,
                split=config.DATASET.SPLIT,
                modality="flow",
                train_mode=False,
                sample_frames_at_test=False,
                transform=torchvision.transforms.Compose([
                    GroupScale(config.TRAIN.RESIZE_MIN),
                    GroupCenterCrop(config.TRAIN.INPUT_SIZE),
                    GroupNormalize(modality="flow"),
                    Stack(),
                ])
            ),
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        flow_model_file = config.TEST.MODEL_FLOW
        if not os.path.exists(flow_model_file):
            raise FileNotFoundError(flow_model_file, " does not exist")
        flow_model = load_model(modality="flow", state_dict_file=flow_model_file)

        print("scoring with flow model")
        targets, flow_predictions = test(flow_model, flow_loader, "flow")

        del flow_model

        targets = targets.cuda(non_blocking=True)
        flow_top1_accuracy = accuracy(flow_predictions, targets, topk=(1, ))
        print("flow top1 accuracy: ", flow_top1_accuracy[0].cpu().numpy().tolist())

    if config.TEST.MODALITY == "combined":
        predictions = torch.stack([rgb_predictions, flow_predictions])
        predictions_mean = torch.mean(predictions, dim=0)
        top1accuracy = accuracy(predictions_mean, targets, topk=(1, ))
        print("combined top1 accuracy: ", top1accuracy[0].cpu().numpy().tolist())


if __name__ == "__main__":
    fire.Fire(run)
