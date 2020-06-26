# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import numpy as np
import os
import os.path as osp
from typing import Dict, List
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from .bbox import TrackingBbox
from .references.fairmot.datasets.dataset.jde import JointDataset
from .opts import opts
from ..common.gpu import db_num_workers


class TrackingDataset:
    """A multi-object tracking dataset."""

    def __init__(
        self, data_root: str, name: str = "default", batch_size: int = 12,
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
        
def boxes_to_mot(results: Dict[int, List[TrackingBbox]]) -> None:
    """
    Save the predicted tracks to csv file in MOT challenge format ["frame", "id", "left", "top", "width", "height",]
    
    Args:
        results: dictionary mapping frame id to a list of predicted TrackingBboxes
        txt_path: path to which results are saved in csv file
    
    """   
    # convert results to dataframe in MOT challenge format
    preds = OrderedDict(sorted(results.items()))
    bboxes = [
        [
            bb.frame_id,
            bb.track_id,
            bb.top,
            bb.left,
            bb.bottom - bb.top,
            bb.right - bb.left,
            1, -1, -1, -1,
        ]
        for _, v in preds.items()
        for bb in v
    ]
    bboxes_formatted = np.array(bboxes)
    
    return bboxes_formatted