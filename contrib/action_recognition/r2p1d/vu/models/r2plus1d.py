# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import os
import time
import warnings

try:
    from apex import amp
    AMP_AVAILABLE = True
except ModuleNotFoundError:
    AMP_AVAILABLE = False
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vu.utils import Config
from vu.data import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    show_batch as _show_batch,
    VideoDataset,
)

from vu.utils.metrics import accuracy, AverageMeter

# From https://github.com/moabitcoin/ig65m-pytorch
TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"
MODELS = {
    # model: output classes
    'r2plus1d_34_32_ig65m': 359,
    'r2plus1d_34_32_kinetics': 400,
    'r2plus1d_34_8_ig65m': 487,
    'r2plus1d_34_8_kinetics': 400,
}


class R2Plus1D(object):
    def __init__(self, cfgs):
        self.configs = Config(cfgs)
        self.train_ds, self.valid_ds = self.load_datasets(self.configs)
        self.model = self.init_model(
            self.configs.sample_length,
            self.configs.base_model,
            self.configs.num_classes
        )
        self.model_name = "r2plus1d_34_{}_{}".format(self.configs.sample_length, self.configs.base_model)

    @staticmethod
    def init_model(sample_length, base_model, num_classes=None):
        if sample_length not in (8, 32):
            raise ValueError(
                "Not supported input frame length {}. Should be 8 or 32"
                .format(sample_length)
            )
        if base_model not in ('ig65m', 'kinetics'):
            raise ValueError(
                "Not supported model {}. Should be 'ig65m' or 'kinetics'"
                .format(base_model)
            )

        model_name = "r2plus1d_34_{}_{}".format(sample_length, base_model)

        print("Loading {} model".format(model_name))

        model = torch.hub.load(
            TORCH_R2PLUS1D, model_name, num_classes=MODELS[model_name], pretrained=True
        )

        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @staticmethod
    def load_datasets(cfgs):
        """Load VideoDataset

        Args:
            cfgs (dict or Config): Dataset configuration. For validation dataset,
                data augmentation such as random shift and temporal jitter is not used.

        Return:
             VideoDataset, VideoDataset: Train and validation datasets.
                If split file is not provided, returns None.
        """
        cfgs = Config(cfgs)

        train_split = cfgs.get('train_split', None)
        train_ds = None if train_split is None else VideoDataset(
            split_file=train_split,
            video_dir=cfgs.video_dir,
            num_segments=1,
            sample_length=cfgs.sample_length,
            sample_step=cfgs.get('temporal_jitter_step', cfgs.get('sample_step', 1)),
            input_size=112,
            im_scale=cfgs.get('im_scale', 128),
            resize_keep_ratio=cfgs.get('resize_keep_ratio', True),
            mean=cfgs.get('mean', DEFAULT_MEAN),
            std=cfgs.get('std', DEFAULT_STD),
            random_shift=cfgs.get('random_shift', True),
            temporal_jitter=True if cfgs.get('temporal_jitter_step', 0) > 0 else False,
            flip_ratio=cfgs.get('flip_ratio', 0.5),
            random_crop=cfgs.get('random_crop', True),
            random_crop_scales=cfgs.get('random_crop_scales', (0.6, 1.0)),
            video_ext=cfgs.video_ext,
        )

        valid_split = cfgs.get('valid_split', None)
        valid_ds = None if valid_split is None else VideoDataset(
            split_file=valid_split,
            video_dir=cfgs.video_dir,
            num_segments=1,
            sample_length=cfgs.sample_length,
            sample_step=cfgs.get('sample_step', 1),
            input_size=112,
            im_scale=cfgs.get('im_scale', 128),
            resize_keep_ratio=True,
            mean=cfgs.get('mean', DEFAULT_MEAN),
            std=cfgs.get('std', DEFAULT_STD),
            random_shift=False,
            temporal_jitter=False,
            flip_ratio=0.0,
            random_crop=False,  # == Center crop
            random_crop_scales=None,
            video_ext=cfgs.video_ext,
        )

        return train_ds, valid_ds

    def show_batch(self, which_data='train', num_samples=1):
        """Plot first few samples in the datasets"""
        if which_data == 'train':
            batch = [self.train_ds[i][0] for i in range(num_samples)]
        elif which_data == 'valid':
            batch = [self.valid_ds[i][0] for i in range(num_samples)]
        else:
            raise ValueError("Unknown data type {}".format(which_data))
        _show_batch(
            batch,
            self.configs.sample_length,
            mean=self.configs.get('mean', DEFAULT_MEAN),
            std=self.configs.get('std', DEFAULT_STD),
        )

    def freeze(self):
        """Freeze model except the last layer"""
        self._set_requires_grad(False)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad=True):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def fit(self, train_cfgs):
        train_cfgs = Config(train_cfgs)

        model_dir = train_cfgs.get('model_dir', "checkpoints")
        os.makedirs(model_dir, exist_ok=True)

        if cuda.is_available():
            device = torch.device("cuda")
            num_devices = cuda.device_count()
            # Look for the optimal set of algorithms to use in cudnn. Use this only with fixed-size inputs.
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            num_devices = 1

        data_loaders = {}
        if self.train_ds is not None:
            data_loaders['train'] = DataLoader(
                self.train_ds,
                batch_size=train_cfgs.get('batch_size', 8) * num_devices,
                shuffle=True,
                num_workers=0,  # Torch 1.2 has a bug when num-workers > 0 (0 means run a main-processor worker)
                pin_memory=True,
            )
        if self.valid_ds is not None:
            data_loaders['valid'] = DataLoader(
                self.valid_ds,
                batch_size=train_cfgs.get('batch_size', 8) * num_devices,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        # Move model to gpu before constructing optimizers and amp.initialize
        self.model.to(device)

        named_params_to_update = {}
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print("\t{}".format(name))

        optimizer = optim.SGD(
            list(named_params_to_update.values()),
            lr=train_cfgs.lr,
            momentum=train_cfgs.momentum,
            weight_decay=train_cfgs.weight_decay,
        )

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if train_cfgs.get('mixed_prec', False) and AMP_AVAILABLE:
            # 'O0': Full FP32, 'O1': Conservative, 'O2': Standard, 'O3': Full FP16
            self.model, optimizer = amp.initialize(
                self.model,
                optimizer,
                opt_level="O1",
                loss_scale="dynamic",
                # keep_batchnorm_fp32=True doesn't work on 'O1'
            )

        # Learning rate scheduler
        scheduler = None
        warmup_pct = train_cfgs.get('warmup_pct', None)
        lr_decay_steps = train_cfgs.get('lr_decay_steps', None)
        if warmup_pct is not None:
            # Use warmup with the one-cycle policy
            lr_decay_total_steps = train_cfgs.epochs if lr_decay_steps is None else lr_decay_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=train_cfgs.lr,
                total_steps=lr_decay_total_steps,
                pct_start=train_cfgs.get('warmup_pct', 0.3),
                base_momentum=0.9*train_cfgs.momentum,
                max_momentum=train_cfgs.momentum,
                final_div_factor=1/train_cfgs.get('lr_decay_factor', 0.0001),
            )
        elif lr_decay_steps is not None:
            lr_decay_total_steps = train_cfgs.epochs
            # Simple step-decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_decay_steps,
                gamma=train_cfgs.get('lr_decay_factor', 0.1),
            )

        # DataParallel after amp.initialize
        if num_devices > 1:
            model = nn.DataParallel(self.model)
        else:
            model = self.model

        criterion = nn.CrossEntropyLoss().to(device)

        for e in range(1, train_cfgs.epochs + 1):
            print("Epoch {} ==========".format(e))
            if scheduler is not None:
                print("lr={}".format(scheduler.get_lr()))
            
            self.train_an_epoch(
                model,
                data_loaders,
                device,
                criterion,
                optimizer,
                grad_steps=train_cfgs.grad_steps,
                mixed_prec=train_cfgs.mixed_prec,
            )
            if scheduler is not None and e < lr_decay_total_steps:
                scheduler.step()
                
            self.save(
                os.path.join(
                    model_dir,
                    "{model_name}_{epoch}.pt".format(
                        model_name=train_cfgs.get('model_name', self.model_name),
                        epoch=str(e).zfill(3)
                    )
                )
            )

    @staticmethod
    def train_an_epoch(
        model,
        data_loaders,
        device,
        criterion,
        optimizer,
        grad_steps=1,
        mixed_prec=False,
    ):
        """Train / validate a model for one epoch.

        :param model:
        :param data_loaders: dict {'train': train_dl, 'valid': valid_dl}
        :param device:
        :param criterion:
        :param optimizer:
        :param grad_steps: If > 1, use gradient accumulation. Useful for larger batching
        :param mixed_prec: If True, use FP16 + FP32 mixed precision via NVIDIA apex.amp
        :return: dict {
            'train/time': batch_time.avg,
            'train/loss': losses.avg,
            'train/top1': top1.avg,
            'train/top5': top5.avg,
            'valid/time': ...
        }
        """
        assert "train" in data_loaders
        if mixed_prec and not AMP_AVAILABLE:
            warnings.warn(
                "NVIDIA apex module is not installed. Cannot use mixed-precision."
            )

        result = OrderedDict()
        for phase in ["train", "valid"]:
            # switch mode
            if phase == "train":
                model.train()
            else:
                model.eval()

            dl = data_loaders[phase]

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            # top5 = AverageMeter()

            end = time.time()
            for step, (inputs, target) in enumerate(dl, start=1):
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    # compute output
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    # measure accuracy and record loss
                    prec1 = accuracy(outputs, target)

                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1[0], inputs.size(0))
                    # top5.update(prec5[0], inputs.size(0))

                    if phase == "train":
                        # make the accumulated gradient to be the same scale as without the accumulation
                        loss = loss / grad_steps

                        if mixed_prec and AMP_AVAILABLE:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        if step % grad_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

            print(
                "{} took {:.2f} sec: loss = {:.4f}, top1_acc = {:.4f}".format(
                    phase, batch_time.sum, losses.avg, top1.avg.item()
                )
            )
            result["{}/time".format(phase)] = batch_time.sum
            result["{}/loss".format(phase)] = losses.avg
            result["{}/top1".format(phase)] = top1.avg
            # result["{}/top5".format(phase)] = top5.avg

        return result

    def save(self, model_path):
        torch.save(
            self.model.state_dict(),
            model_path
        )

    def load(self, model_name, model_dir="checkpoints"):
        """
        TODO accept epoch. If None, load the latest model.
        :param model_name: Model name format should be 'name_0EE' where E is the epoch
        :param model_dir: By default, 'checkpoints'
        :return:
        """
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "{}.pt".format(model_name))
        ))
