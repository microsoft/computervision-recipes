# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import os
import time
import warnings
from typing import Union
from pathlib import Path

try:
    from apex import amp

    AMP_AVAILABLE = True
except ModuleNotFoundError:
    AMP_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from ..common.misc import Config
from ..common.gpu import torch_device, num_devices
from .dataset import VideoDataset

from .references.metrics import accuracy, AverageMeter

# These paramaters are set so that we can use torch hub to download pretrained
# models from the specified repo
TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"  # From https://github.com/moabitcoin/ig65m-pytorch
MODELS = {
    # Model name followed by the number of output classes.
    "r2plus1d_34_32_ig65m": 359,
    "r2plus1d_34_32_kinetics": 400,
    "r2plus1d_34_8_ig65m": 487,
    "r2plus1d_34_8_kinetics": 400,
}


class VideoLearner(object):
    """ Video recognition learner object that handles training loop and evaluation. """

    def __init__(
        self,
        dataset: VideoDataset,
        num_classes: int,  # ie 51 for hmdb51
        base_model: str = "ig65m",  # or "kinetics"
    ) -> None:
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
        self.model, self.model_name = self.init_model(
            self.dataset.sample_length, base_model, num_classes,
        )

    @staticmethod
    def init_model(
        sample_length: int, base_model: str, num_classes: int = None
    ) -> torchvision.models.video.resnet.VideoResNet:
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
                f"Not supported model {base_model}. Should be 'ig65m' or 'kinetics'"
            )

        # Decide if to use pre-trained weights for DNN trained using 8 or for 32 frames
        if sample_length <= 8:
            model_sample_length = 8
        else:
            model_sample_length = 32

        model_name = f"r2plus1d_34_{model_sample_length}_{base_model}"

        print(f"Loading {model_name} model")

        model = torch.hub.load(
            TORCH_R2PLUS1D,
            model_name,
            num_classes=MODELS[model_name],
            pretrained=True,
        )

        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model, model_name

    def freeze(self) -> None:
        """Freeze model except the last layer"""
        self._set_requires_grad(False)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad=True) -> None:
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def fit(self, train_cfgs) -> None:
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
                print(f"\t{name}")

        # create optimizer
        momentum = train_cfgs.get("momentum", 0.95)
        optimizer = optim.SGD(
            list(named_params_to_update.values()),
            lr=train_cfgs.lr,
            momentum=momentum,
            weight_decay=train_cfgs.get("weight_decay", 0.0001),
        )

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if train_cfgs.get("mixed_prec", False):
            # break if not AMP_AVAILABLE
            assert AMP_AVAILABLE
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
            nn.DataParallel(self.model) if count_devices > 1 else self.model
        )

        criterion = nn.CrossEntropyLoss().to(device)

        for e in range(1, train_cfgs.epochs + 1):
            print(f"Epoch {e} ==========")
            print(f"lr={scheduler.get_lr()}")

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
    ) -> None:
        """Train / validate a model for one epoch.

        Args:
            model: the model to use to train
            data_loaders: dict {'train': train_dl, 'valid': valid_dl}
            device: gpu or not
            criterion: TODO
            optimizer: TODO
            grad_steps: If > 1, use gradient accumulation. Useful for larger batching
            mixed_prec: If True, use FP16 + FP32 mixed precision via NVIDIA apex.amp

        Return:
            dict {
                'train/time': batch_time.avg,
                'train/loss': losses.avg,
                'train/top1': top1.avg,
                'train/top5': top5.avg,
                'valid/time': ...
            }
        """
        if mixed_prec and not AMP_AVAILABLE:
            warnings.warn(
                """
                NVIDIA apex module is not installed. Cannot use
                mixed-precision. Turning off mixed-precision.
                """
            )
            mixed_prec = False

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

                        if mixed_prec:
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
                f"{phase} took {batch_time.sum:.2f} sec: loss = {losses.avg:.4f}, top1_acc = {top1.avg:.4f}, top5_acc = {top5.avg:.4f}"
            )
            result[f"{phase}/time"] = batch_time.sum
            result[f"{phase}/loss"] = losses.avg
            result[f"{phase}/top1"] = top1.avg
            result[f"{phase}/top5"] = top5.avg

        return result

    def save(self, model_path: Union[Path, str]) -> None:
        """ Save the model to a path on disk. """
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_name: str, model_dir: str = "checkpoints") -> None:
        """
        TODO accept epoch. If None, load the latest model.
        :param model_name: Model name format should be 'name_0EE' where E is the epoch
        :param model_dir: By default, 'checkpoints'
        :return:
        """
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, f"{model_name}.pt"))
        )
