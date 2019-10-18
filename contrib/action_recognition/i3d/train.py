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
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from default import _C as config
from default import update_config

from videotransforms import (
    GroupRandomCrop, GroupRandomHorizontalFlip,
    GroupScale, GroupCenterCrop, GroupNormalize, Stack
)
from models.pytorch_i3d import InceptionI3d
from metrics import accuracy, AverageMeter
from dataset import I3DDataSet 


# to work with vscode debugger https://github.com/joblib/joblib/issues/864
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def train(train_loader, model, criterion, optimizer, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        output = torch.mean(output, dim=2)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        loss = loss / config.TRAIN.GRAD_ACCUM_STEPS
        
        loss.backward()

        if step % config.TRAIN.GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.TRAIN.PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
        
        if writer:
            writer.add_scalar('train/loss', losses.avg, epoch+1)
            writer.add_scalar('train/top1', top1.avg, epoch+1)
            writer.add_scalar('train/top5', top5.avg, epoch+1)


def validate(val_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=2)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1,5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.TEST.PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    step, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=top1, top5=top5, loss=losses)))

        if writer:
            writer.add_scalar('val/loss', losses.avg, epoch+1)
            writer.add_scalar('val/top1', top1.avg, epoch+1)
            writer.add_scalar('val/top5', top5.avg, epoch+1)

    return losses.avg


def run(*options, cfg=None):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options loaded from default.py will be overridden by options loaded from cfg file
        Options passed in through options argument will override option loaded from cfg file
    
    Args:
        *options (str,int ,optional): Options used to overide what is loaded from the config. 
                                      To see what options are available consult default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
    """
    update_config(config, options=options, config_file=cfg)

    print("Training ", config.TRAIN.MODALITY, " model.")
    print("Batch size:", config.TRAIN.BATCH_SIZE, " Gradient accumulation steps:", config.TRAIN.GRAD_ACCUM_STEPS)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Log to tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Setup dataloaders
    train_loader = torch.utils.data.DataLoader(
        I3DDataSet(
            data_root=config.DATASET.DIR,
            split=config.DATASET.SPLIT,
            sample_frames=config.TRAIN.SAMPLE_FRAMES,
            modality=config.TRAIN.MODALITY,
            transform=torchvision.transforms.Compose([
                GroupScale(config.TRAIN.RESIZE_MIN),
                GroupRandomCrop(config.TRAIN.INPUT_SIZE),
                GroupRandomHorizontalFlip(),
                GroupNormalize(modality=config.TRAIN.MODALITY),
                Stack(),
            ])
        ),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = torch.utils.data.DataLoader(
        I3DDataSet(
            data_root=config.DATASET.DIR,
            split=config.DATASET.SPLIT,
            modality=config.TRAIN.MODALITY,
            train_mode=False,
            transform=torchvision.transforms.Compose([
                GroupScale(config.TRAIN.RESIZE_MIN),
                GroupCenterCrop(config.TRAIN.INPUT_SIZE),
                GroupNormalize(modality=config.TRAIN.MODALITY),
                Stack(),
            ]),
        ),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Setup model
    if config.TRAIN.MODALITY == "RGB":
        channels = 3
        checkpoint = config.MODEL.PRETRAINED_RGB
    elif config.TRAIN.MODALITY == "flow":
        channels = 2
        checkpoint = config.MODEL.PRETRAINED_FLOW
    else:
        raise ValueError("Modality must be RGB or flow")

    i3d_model = InceptionI3d(400, in_channels=channels)
    i3d_model.load_state_dict(torch.load(checkpoint))

    # Replace final FC layer to match dataset
    i3d_model.replace_logits(config.DATASET.NUM_CLASSES)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
       i3d_model.parameters(), 
       lr=0.1,
       momentum=0.9, 
       weight_decay=0.0000001
    )

    i3d_model = i3d_model.cuda()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=2,
        verbose=True,
        threshold=1e-4,
        min_lr=1e-4
    )

    # Data-parallel
    devices_lst = list(range(torch.cuda.device_count()))
    print("Devices {}".format(devices_lst))
    if len(devices_lst) > 1:
        i3d_model = torch.nn.DataParallel(i3d_model)

    if not os.path.exists(config.MODEL.CHECKPOINT_DIR):
        os.makedirs(config.MODEL.CHECKPOINT_DIR)
    
    for epoch in range(config.TRAIN.MAX_EPOCHS):

        train(train_loader,
            i3d_model,
            criterion,
            optimizer,
            epoch,
            writer
        )

        if (epoch + 1) % config.TEST.EVAL_FREQ == 0 or epoch == config.TRAIN.MAX_EPOCHS - 1:
            val_loss = validate(val_loader, i3d_model, criterion, epoch, writer)
            scheduler.step(val_loss)
            torch.save(
                i3d_model.module.state_dict(),
                config.MODEL.CHECKPOINT_DIR+'/'+config.MODEL.NAME+'_split'+str(config.DATASET.SPLIT)+'_epoch'+str(epoch).zfill(3)+'.pt'
            )

    writer.close()


if __name__ == "__main__":
    fire.Fire(run)