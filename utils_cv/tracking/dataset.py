# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import os.path as osp
from typing import Dict
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from .references.fairmot.datasets.dataset.jde import JointDataset
from .opts import opts
from ..common.gpu import db_num_workers


class TrackingDataset:
    """A multi-object tracking dataset."""

    def __init__(
        self,
        data_root: str,
        name: str = "default",
        batch_size: int = 12,
    ) -> None:
        """
        Args:
            data_root: root data directory containing image and annotation subdirectories
            name: user-friendly name for the dataset
            batch_size: batch size
        """
        transforms = T.Compose([T.ToTensor()])
        self.batch_size = batch_size

        opt = opts()

        train_list_path = osp.join(data_root, "{}.train".format(name))
        with open(train_list_path, "a") as f:
            for im_name in sorted(os.listdir(osp.join(data_root, "images"))):
                f.write(osp.join("images", im_name) + "\n")

        self.train_data = JointDataset(
            opt.opt,
            data_root,
            {name: train_list_path},
            (opt.input_w, opt.input_h),
            augment=True,
            transforms=transforms,
        )

        self._init_dataloaders()

    def _init_dataloaders(self) -> None:
        """ Create training dataloader """
        self.train_dl = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=db_num_workers(),
            pin_memory=True,
            drop_last=True,
        )
