# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
from functools import partial
import os
import os.path as osp
from pathlib import Path
import random
import tempfile
from typing import Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from .bbox import TrackingBbox
from .opts import opts
from .references.fairmot.datasets.dataset.jde import JointDataset
from ..common.gpu import db_num_workers
from ..detection.dataset import parse_pascal_voc_anno
from ..detection.plot import plot_detections, plot_grid


class TrackingDataset:
    """A multi-object tracking dataset."""

    def __init__(
        self,
        root: str,
        name: str = "default",
        batch_size: int = 12,
        im_dir: str = "images",
        anno_dir: str = "annotations",
    ) -> None:
        """
        Args:
            data_root: root data directory containing image and annotation subdirectories
            name: user-friendly name for the dataset
            batch_size: batch size
            anno_dir: the name of the annotation subfolder under the root directory
            im_dir: the name of the image subfolder under the root directory.
        """
        self.root = root
        self.name = name
        self.batch_size = batch_size
        self.im_dir = Path(im_dir)
        self.anno_dir = Path(anno_dir)

        # set these to None so taht can use the 'plot_detections' function
        self.keypoints = None
        self.mask_paths = None

        # Init FairMOT opt object with all parameter settings
        opt = opts()

        # Read annotations
        self._read_annos()

        # Save annotation in FairMOT format
        self._write_fairMOT_format()

        # Create FairMOT dataset object
        transforms = T.Compose([T.ToTensor()])
        self.train_data = JointDataset(
            opt,
            self.root,
            {name: self.fairmot_imlist_path},
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

    def _read_annos(self) -> None:
        """ Parses all Pascal VOC formatted annotation files to extract all
        possible labels. """
        # All annotation files are assumed to be in the anno_dir directory,
        # and images in the im_dir directory
        self.im_filenames = sorted(os.listdir(self.root / self.im_dir))
        im_paths = [
            os.path.join(self.root / self.im_dir, s) for s in self.im_filenames
        ]
        anno_filenames = [
            os.path.splitext(s)[0] + ".xml" for s in self.im_filenames
        ]

        # Read all annotations
        self.im_paths = []
        self.anno_paths = []
        self.anno_bboxes = []
        for anno_idx, anno_filename in enumerate(anno_filenames):
            anno_path = self.root / self.anno_dir / str(anno_filename)

            # Parse annotation file
            anno_bboxes, _, _ = parse_pascal_voc_anno(anno_path)

            # Store annotation info
            self.im_paths.append(im_paths[anno_idx])
            self.anno_paths.append(anno_path)
            self.anno_bboxes.append(anno_bboxes)
        assert len(self.im_paths) == len(self.anno_paths)

        # Get list of all labels
        labels = []
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                if anno_bbox.label_name is not None:
                    labels.append(anno_bbox.label_name)
        self.labels = list(set(labels))

        # Set for each bounding box label name also what its integer representation is
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                if anno_bbox.label_name is None:
                    # background rectangle is assigned id 0 by design
                    anno_bbox.label_idx = 0
                else:
                    label = self.labels.index(anno_bbox.label_name) + 1
                    anno_bbox.label_idx = label

        # Get image sizes. Note that Image.open() only loads the image header,
        # not the full images and is hence fast.
        self.im_sizes = np.array([Image.open(p).size for p in self.im_paths])

    def _write_fairMOT_format(self) -> None:
        """ Write bounding box information in the format FairMOT expects for training."""
        fairmot_annos_dir = os.path.join(self.root, "labels_with_ids")
        os.makedirs(fairmot_annos_dir, exist_ok=True)

        # Create for each image a annotation .txt file in FairMOT format
        for filename, bboxes, im_size in zip(
            self.im_filenames, self.anno_bboxes, self.im_sizes
        ):
            im_width = float(im_size[0])
            im_height = float(im_size[1])
            fairmot_anno_path = os.path.join(
                fairmot_annos_dir, filename[:-4] + ".txt"
            )

            with open(fairmot_anno_path, "w") as f:
                for bbox in bboxes:
                    tid_curr = bbox.label_idx - 1
                    x = round(bbox.left + bbox.width() / 2.0)
                    y = round(bbox.top + bbox.height() / 2.0)
                    w = bbox.width()
                    h = bbox.height()

                    label_str = "0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        tid_curr,
                        x / im_width,
                        y / im_height,
                        w / im_width,
                        h / im_height,
                    )
                    f.write(label_str)

        # write all image filenames into a <name>.train file required by FairMOT
        self.fairmot_imlist_path = osp.join(
            self.root, "{}.train".format(self.name)
        )
        with open(self.fairmot_imlist_path, "w") as f:
            for im_filename in sorted(self.im_filenames):
                f.write(osp.join(self.im_dir, im_filename) + "\n")

    def show_ims(self, rows: int = 1, cols: int = 3, seed: int = None) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display
            cols: cols to display, NOTE: use 3 for best looking grid
            seed: random seed for selecting images

        Returns None but displays a grid of annotated images.
        """
        if seed:
            random.seed(seed or self.seed)

        def helper(im_paths):
            idx = random.randrange(len(im_paths))
            detection = {
                "idx": idx,
                "im_path": im_paths[idx],
                "det_bboxes": [],
            }
            return detection, self, None, None

        plot_grid(
            plot_detections,
            partial(helper, self.im_paths),
            rows=rows,
            cols=cols,
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
            bb.frame_id + 1,
            bb.track_id,
            bb.left,
            bb.top,
            bb.right - bb.left,
            bb.bottom - bb.top,
            1,
            -1,
            -1,
            -1,
        ]
        for _, v in preds.items()
        for bb in v
    ]
    bboxes_formatted = np.array(bboxes)

    return bboxes_formatted
