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

from IPython.core.debugger import set_trace
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim

from ..common.misc import Config
from ..common.gpu import torch_device, num_devices
from .dataset import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    VideoDataset,
    get_default_tfms_config,
)

from .references.metrics import accuracy, AverageMeter

# From https://github.com/moabitcoin/ig65m-pytorch
TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"
MODELS = {
    # model: output classes
    "r2plus1d_34_32_ig65m": 359,
    "r2plus1d_34_32_kinetics": 400,
    "r2plus1d_34_8_ig65m": 487,
    "r2plus1d_34_8_kinetics": 400,
}


class VideoLearner(object):
    """ tk """

    def __init__(
        self,
        dataset: VideoDataset,
        num_classes: int,  # ie 51 for hmdb51
        base_model: str = "ig65m",  # or "kinetics"
    ):
        """ By default, the Video Learner will use a R2plus1D model. Pass in
        a dataset of type Video Dataset and the Video Learner will intialize
        the model.

        Args:
            dataset: the datset to use for this model
            num_class: the number of actions/classifications
            base_model: the R2plus1D model is based on either ig65m or
            kinetics. By default it will use the weights from ig65m since it
            tends attain higher results.
        """
        self.dataset = dataset
        self.model = self.init_model(
            self.dataset.sample_length, base_model, num_classes,
        )
        self.model_name = "r2plus1d_34_{}_{}".format(
            self.dataset.sample_length, base_model
        )

    @staticmethod
    def init_model(
        sample_length: int, base_model: str, num_classes: int = None
    ):
        """
        Initializes the model by loading it using torch's `hub.load`
        functionality. Uses the model from TORCH_R2PLUS1D.

        Args:
            sample_length: Number of consecutive frames to sample from a video (i.e. clip length).
            base_model: the R2plus1D model is based on either ig65m or kinetics.
            num_classes: the number of classes/actions

        Returns:
            Load a model from a github repo, with pretrained weights
        """
        if base_model not in ("ig65m", "kinetics"):
            raise ValueError(
                "Not supported model {}. Should be 'ig65m' or 'kinetics'".format(
                    base_model
                )
            )

        # Decide if to use pre-trained weights for DNN trained using 8 or for 32 frames
        if sample_length <= 8:
            model_sample_length = 8
        else:
            model_sample_length = 32
        model_name = "r2plus1d_34_{}_{}".format(
            model_sample_length, base_model
        )

        print("Loading {} model".format(model_name))

        model = torch.hub.load(
            TORCH_R2PLUS1D,
            model_name,
            num_classes=MODELS[model_name],
            pretrained=True,
        )

        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

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
        """ The primary fit function """
        train_cfgs = Config(train_cfgs)

        model_dir = train_cfgs.get("model_dir", "checkpoints")
        os.makedirs(model_dir, exist_ok=True)

        data_loaders = {}
        data_loaders["train"] = self.dataset.train_dl
        data_loaders["valid"] = self.dataset.test_dl

        # Move model to gpu before constructing optimizers and amp.initialize
        device = torch_device()
        self.model.to(device)
        count_devices = num_devices()
        torch.backends.cudnn.benchmark = True

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

        momentum = train_cfgs.get("momentum", 0.95)
        optimizer = optim.SGD(
            list(named_params_to_update.values()),
            lr=train_cfgs.lr,
            momentum=momentum,
            weight_decay=train_cfgs.get("weight_decay", 0.0001),
        )

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if train_cfgs.get("mixed_prec", False) and AMP_AVAILABLE:
            # 'O0': Full FP32, 'O1': Conservative, 'O2': Standard, 'O3': Full FP16
            self.model, optimizer = amp.initialize(
                self.model,
                optimizer,
                opt_level="O1",
                loss_scale="dynamic",
                # keep_batchnorm_fp32=True doesn't work on 'O1'
            )

        # Learning rate scheduler
        if train_cfgs.get("use_one_cycle_policy", False):
            # Use warmup with the one-cycle policy
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=train_cfgs.lr,
                total_steps=train_cfgs.epochs,
                pct_start=train_cfgs.get("warmup_pct", 0.3),
                base_momentum=0.9 * momentum,
                max_momentum=momentum,
            )
        else:
            # Simple step-decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_cfgs.get("lr_step_size", float("inf")),
                gamma=train_cfgs.get("lr_gamma", 0.1),
            )

        # DataParallel after amp.initialize
        model = (
            nn.DataParallel(self.model)
            if count_devices > 1
            else model = self.model
        )

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

            scheduler.step()

            if train_cfgs.get("save_models", False):
                self.save(
                    os.path.join(
                        model_dir,
                        "{model_name}_{epoch}.pt".format(
                            model_name=train_cfgs.get(
                                "model_name", self.model_name
                            ),
                            epoch=str(e).zfill(3),
                        ),
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
            top5 = AverageMeter()

            end = time.time()
            for step, (inputs, target) in enumerate(dl, start=1):
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    # compute output
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, target, topk=(1, 5))

                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1[0], inputs.size(0))
                    top5.update(prec5[0], inputs.size(0))

                    if phase == "train":
                        # make the accumulated gradient to be the same scale as without the accumulation
                        loss = loss / grad_steps

                        if mixed_prec and AMP_AVAILABLE:
                            with amp.scale_loss(
                                loss, optimizer
                            ) as scaled_loss:
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
                "{} took {:.2f} sec: loss = {:.4f}, top1_acc = {:.4f}, top5_acc = {:.4f}".format(
                    phase, batch_time.sum, losses.avg, top1.avg, top5.avg
                )
            )
            result["{}/time".format(phase)] = batch_time.sum
            result["{}/loss".format(phase)] = losses.avg
            result["{}/top1".format(phase)] = top1.avg
            result["{}/top5".format(phase)] = top5.avg

        return result

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_name, model_dir="checkpoints"):
        """
        TODO accept epoch. If None, load the latest model.
        :param model_name: Model name format should be 'name_0EE' where E is the epoch
        :param model_dir: By default, 'checkpoints'
        :return:
        """
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "{}.pt".format(model_name)))
        )
