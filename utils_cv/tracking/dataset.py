# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from .references.fairmot.datasets.dataset.jde import JointDataset
from ..common.gpu import db_num_workers


class TrackingDataset:
    """A multi-object tracking dataset."""

    def __init__(
        self, data_root: str, train_paths: Dict, batch_size: int = 12
    ) -> None:
        """
        Args:
            data_root: root data directory
            train_paths: dictionary of paths defining training data
            batch_size: batch size

        Note: the path dictionaries map user-friendly dataset name to a filename, e.g. {"custom": "MyCustomDataset.train"}
            The file is a raw text file listing the sequence of image paths. Multiple sequences (videos) can be listed,
            and the sequences must be in order. 
        
        TODO: train-test split
        """
        transforms = T.Compose([T.ToTensor()])
        self.batch_size = batch_size

        self.train_data = JointDataset(
            opt,
            data_root,
            train_paths,
            im_size,
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