# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from copy import deepcopy
import glob
import requests
import os
import os.path as osp
from typing import Dict, List, Tuple

import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from .references.fairmot.datasets.dataset.jde import LoadImages, LoadVideo
from .references.fairmot.models.model import (
    create_model,
    load_model,
    save_model,
)
from .references.fairmot.tracker.multitracker import JDETracker
from .references.fairmot.trains.train_factory import train_factory

from .bbox import TrackingBbox
from .dataset import TrackingDataset
from .opts import opts
from ..common.gpu import torch_device

BASELINE_URL = (
    "https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu"
)


def _download_baseline(url, destination) -> None:
    """
    Download the baseline model .pth file to the destination.

    Args:
        url: a Google Drive url of the form "https://drive.google.com/open?id={id}"
        destination: path to save the model to

    Implementation based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    session = requests.Session()
    id = url.split("id=")[-1]
    response = session.get(url, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(
            url, params={"id": id, "confirm": token}, stream=True
        )

    save_response_content(response, destination)


def _get_gpu_str():
    if cuda.is_available():
        devices = [str(x) for x in range(cuda.device_count())]
        return ",".join(devices)
    else:
        return "-1"  # cpu


class TrackingLearner(object):
    """Tracking Learner for Multi-Object Tracking"""

    def __init__(
        self,
        dataset: TrackingDataset,
        model: nn.Module = None,
        arch: str = "dla_34",
        head_conv: int = None,
    ) -> None:
        """
        Initialize learner object.

        Defaults to the FairMOT model.

        Args:
            dataset: the dataset
            model: the model
            arch: the model architecture
                Supported architectures: resdcn_34, resdcn_50, resfpndcn_34, dla_34, hrnet_32
            head_conv: conv layer channels for output head. None maps to the default setting.
                Set 0 for no conv layer, 256 for resnets, and 256 for dla
        """
        self.opt = opts()
        self.opt.arch = arch
        self.opt.head_conv = head_conv if head_conv else -1
        self.opt.gpus = _get_gpu_str()
        self.opt.device = torch_device()

        self.dataset = dataset
        self.model = model if model is not None else self.init_model()

    def init_model(self) -> nn.Module:
        """
        Download and initialize the baseline FairMOT model.
        """
        model_dir = osp.join(self.opt.root_dir, "models")
        self.model_path = osp.join(model_dir, "all_dla34.pth")

        os.makedirs(model_dir, exist_ok=True)
        _download_baseline(BASELINE_URL, self.model_path)
        return create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)

    def load(self, path, resume=False) -> None:
        self.opt.load_model = path
        self.opt.resume = resume

    def fit(
        self, lr: float = 1e-4, lr_step: str = "20,27", num_epochs: int = 30,
    ) -> None:
        """
        The main training loop.

        Args:
            lr: learning rate for batch size 32
            lr_step: when to drop learning rate by 10
            num_epochs: total training epochs

        Raise:
            Exception if dataset is undefined
        
        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/train.py
        """
        if not self.dataset:
            raise Exception("No dataset provided")

        opt_fit = deepcopy(self.opt)
        opt_fit.lr = lr
        opt_fit.lr_step = lr_step
        opt_fit.num_epochs = num_epochs

        # update dataset options
        opt_fit.update_dataset_info_and_set_heads(self.dataset.train_data)

        # initialize dataloader
        train_loader = self.dataset.train_dl

        self.optimizer = torch.optim.Adam(self.model.parameters(), opt_fit.lr)
        start_epoch = 0
        self.model, self.optimizer, start_epoch = load_model(
            self.model,
            opt_fit.load_model,
            self.optimizer,
            opt_fit.resume,
            opt_fit.lr,
            opt_fit.lr_step,
        )

        Trainer = train_factory[opt_fit.task]
        trainer = Trainer(opt_fit.opt, self.model, self.optimizer)
        trainer.set_device(opt_fit.gpus, opt_fit.chunk_sizes, opt_fit.device)

        # training loop
        for epoch in range(
            start_epoch + 1, start_epoch + opt_fit.num_epochs + 1
        ):
            print(
                "=" * 5,
                f" Epoch: {epoch}/{start_epoch + opt_fit.num_epochs} ",
                "=" * 5,
            )
            self.epoch = epoch
            log_dict_train, _ = trainer.train(epoch, train_loader)
            for k, v in log_dict_train.items():
                print(f"{k}: {v}")
            if epoch in opt_fit.lr_step:
                lr = opt_fit.lr * (0.1 ** (opt_fit.lr_step.index(epoch) + 1))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

    def save(self, path) -> None:
        """
        Save the model to a specified path.
        """
        model_dir, _ = osp.split(path)
        os.makedirs(model_dir, exist_ok=True)

        save_model(path, self.epoch, self.model, self.optimizer)
        print(f"Model saved to {path}")

    def evaluate(self, results, gt) -> pd.DataFrame:
        pass

    def predict(
        self,
        im_or_video_path: str,
        conf_thres: float = 0.6,
        det_thres: float = 0.3,
        nms_thres: float = 0.4,
        track_buffer: int = 30,
        min_box_area: float = 200,
        im_size: Tuple[float, float] = (None, None),
        frame_rate: int = 30,
    ) -> Dict[int, List[TrackingBbox]]:
        """
        Performs inferencing on an image or video path.

        Args:
            im_or_video_path: path to image(s) or video. Supports jpg, jpeg, png, tif formats for images.
                Supports mp4, avi formats for video. 
            conf_thres: confidence thresh for tracking
            det_thres: confidence thresh for detection
            nms_thres: iou thresh for nms
            track_buffer: tracking buffer
            min_box_area: filter out tiny boxes
            input_h: input height. Default from dataset
            input_w: input width. Default from dataset
            frame_rate: frame rate

        Returns a list of TrackingBboxes

        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/track.py
        """
        opt_pred = deepcopy(self.opt)
        opt_pred.conf_thres = conf_thres
        opt_pred.det_thres = det_thres
        opt_pred.nms_thres = nms_thres
        opt_pred.track_buffer = track_buffer
        opt_pred.min_box_area = min_box_area

        input_h, input_w = im_size
        input_height = input_h if input_h else -1
        input_width = input_w if input_w else -1
        opt_pred.update_dataset_res(input_height, input_width)

        # initialize tracker
        tracker = JDETracker(opt_pred.opt, frame_rate=frame_rate)

        # initialize dataloader
        dataloader = self._get_dataloader(
            im_or_video_path, opt_pred.input_h, opt_pred.input_w
        )

        frame_id = 0
        out = {}
        results = []
        for path, img, img0 in dataloader:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_bboxes = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt_pred.min_box_area and not vertical:
                    bb = TrackingBbox(
                        tlbr[1], tlbr[0], tlbr[3], tlbr[2], frame_id, tid
                    )
                    online_bboxes.append(bb)
            out[frame_id] = online_bboxes
            frame_id += 1

        return out

    def _get_dataloader(
        self, im_or_video_path: str, input_h, input_w
    ) -> DataLoader:
        """
        Creates a dataloader from images or video in the given path.

        Args:
            im_or_video_path: path to a root directory of images, or single video or image file.
                Supports jpg, jpeg, png, tif formats for images. Supports mp4, avi formats for video

        Return:
            Dataloader

        Raise:
            Exception if file format is not supported

        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/lib/datasets/dataset/jde.py
        """
        im_format = [".jpg", ".jpeg", ".png", ".tif"]
        video_format = [".mp4", ".avi"]

        # if path is to a root directory of images

        if (
            osp.isdir(im_or_video_path)
            and len(
                list(
                    filter(
                        lambda x: osp.splitext(x)[1].lower() in im_format,
                        sorted(glob.glob("%s/*.*" % im_or_video_path)),
                    )
                )
            )
            > 0
        ):
            return LoadImages(im_or_video_path, img_size=(input_w, input_h))
        # if path is to a single video file
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in video_format
        ):
            return LoadVideo(im_or_video_path, img_size=(input_w, input_h))
        # if path is to a single image file
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in im_format
        ):
            return LoadImages(im_or_video_path, img_size=(input_w, input_h))
        else:
            raise Exception("Image or video format not supported")
