import os
import os.path as osp
import torch
import argparse

from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from .bbox import TrackingBbox
from .opts import opts
from .references.fairmot.datasets.dataset_factory import get_dataset
from .references.fairmot.datasets.dataset.jde import LoadImages, LoadVideo
from .references.fairmot.tracker.multitracker import JDETracker
from .references.fairmot.models.model import (
    create_model,
    load_model,
    save_model,
)
from ..common.gpu import torch_device


class TrackingLearner(object):
    def __init__(
        self,
        root_dir: Path,
        arch: str = "dla_34",
        head_conv: int = -1,
        down_ratio: int = 4,
    ) -> None:
        """
        Initialize learner object.
        Defaults to FairMOT
        """
        self.opt = opts(root_dir).opt
        self.opt.arch = arch
        self.opt.head_conv = head_conv
        self.opt.down_ratio = down_ratio

        # TODO setup logging

    def fit(
        self,
        data_root,  # consider making a custom cvbp Dataset
        data_path,
        im_size,
        lr: float = 1e-4,
        lr_step: str = "20,27",
        num_epochs: int = 30,
        batch_size: int = 12,
        num_iters: int = -1,
        val_intervals: int = 5,
        num_workers: int = 8,
    ) -> None:
        self.opt.lr = lr
        self.opt.lr_step = lr_step
        self.opt.num_epochs = num_epochs
        self.opt.batch_size = batch_size
        self.opt.num_iters = num_iters
        self.opt.val_intervals = val_intervals
        self.opt.num_workers = num_workers
        self.opt.device = torch_device()

        # initialize dataset
        dataset = self._init_dataset(data_root, train_path, im_size)
        self.opt.update_dataset_info_and_set_heads(dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # initialize model
        model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        optimizer = torch.optim.Adam(model.parameters(), self.opt.lr)
        start_epoch = 0
        if self.opt.load_model != "":
            model, optimizer, start_epoch = load_model(
                model,
                self.opt.load_model,
                optimizer,
                self.opt.resume,
                self.opt.lr,
                self.opt.lr_step,
            )

        Trainer = train_factory[self.opt.task]
        trainer = Trainer(self.opt, self.model, optimizer)
        trainer.set_device(
            self.opt.gpus, self.opt.chunk_sizes, self.opt.device
        )
        best = 1e10

        # training loop
        for epoch in range(start_epoch + 1, self.opt.num_epochs + 1):
            mark = epoch if self.opt.save_all else "last"
            log_dict_train, _ = trainer.train(epoch, train_loader)
            ## TODO logging
            if (
                self.opt.val_intervals > 0
                and epoch % self.opt.val_intervals == 0
            ):
                save_model(
                    osp.join(self.opt.save_dir, f"model_{mark}.pth"),
                    epoch,
                    model,
                    optimizer,
                )
            else:
                save_model(
                    osp.join(
                        self.opt.save_dir,
                        f"model_last.pth",
                        epoch,
                        model,
                        optimizer,
                    )
                )

        for epoch in self.opt.lr_step:
            save_model(
                os.path.join(self.opt.save_dir, f"model_{epoch}.pth"),
                epoch,
                model,
                optimizer,
            )
            lr = self.opt.lr * (0.1 ** self.opt.lr_step.index(epoch) + 1)
            ## TODO logging
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if epoch % 5 == 0:  ## TODO make this a param
            save_model(
                osp.join(
                    self.opt.save_dir,
                    f"model_{epoch}.pth",
                    epoch,
                    model,
                    optimizer,
                )
            )

    def _init_dataset(self, data_root, train_path, im_size) -> Dataset:
        Dataset = get_dataset(self.opt.dataset, self.opt.task)
        transforms = T.Compose([T.ToTensor()])
        return Dataset(
            self.opt,
            data_root,
            train_path,
            im_size,
            augment=True,
            transforms=transforms,
        )

    def predict(
        self,
        im_or_video_path: Path,
        load_model: Path = "",  # TODO path to default - coco? baseline all_dla34?
        conf_thres: float = 0.6,
        det_thres: float = 0.3,
        nms_thres: float = 0.4,
        track_buffer: int = 30,
        min_box_area: float = 200,
        input_w: float = -1,
        input_h: float = -1,
        frame_rate: int = 30,
    ) -> Dict[int, List[TrackingBbox]]:

        self.opt.load_model = load_model
        self.opt.conf_thres = conf_thres
        self.opt.det_thres = det_thres
        self.opt.nms_thres = nms_thres
        self.opt.track_buffer = track_buffer
        self.opt.min_box_area = min_box_area
        self.opt.input_w = input_w
        self.opt.input_h = input_h
        self.opt.device = torch_device()

        tracker = JDETracker(self.opt, frame_rate=frame_rate)

        dataloader = self._get_dataloader(im_or_video_path, input_w, input_h)

        frame_id = 0
        out = {}
        results = []
        for path, img, img0 in dataloader:
            # TODO logging
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets = self.tracker.update(blob, img0)
            online_bboxes = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                    bb = TrackingBbox(
                        tlbr[1], tlbr[0], tlbr[3], tlbr[2], frame_id, tid
                    )
                    online_bboxes.append(bb)
            out[frame_id] = online_bboxes
            frame_id += 1

        # TODO add some option to save - in tlbr (consistent with cvbp) or tlwh (consistent with fairmot)?
        return out

    def _get_dataloader(self, im_or_video_path, input_w, input_h) -> DataLoader:
        im_format = [".jpg", ".jpeg", ".png", ".tif"]
        video_format = [".mp4", ".avi"]

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
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in video_format
        ):
            return LoadVideo(im_or_video_path, img_size=(input_w, input_h))
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in im_format
        ):
            return LoadImages(im_or_video_path, img_size=(input_w, input_h))
        else:
            raise Exception("Image or video format not supported.")
