# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from functools import partial
import math
from pathlib import Path
from random import randrange
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image

from .plot import display_bounding_boxes, plot_grid
from .helper import Bbox, Annotation
from .references.utils import collate_fn
from .references.transforms import RandomHorizontalFlip, Compose, ToTensor


def get_transform(train: bool) -> List[object]:
    """ Gets basic the transformations to apply to images.

    Source:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan

    Args:
        train: whether or not we are getting transformations for the training
        set.

    Returns:
        A list of transforms to apply.
    """
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


class DetectionDataset:
    """ An object detection dataset.

    The dunder methods __init__, __getitem__, and __len__ were inspired from code found here:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan
    """

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 2,
        transforms: object = None,
        train_test_ratio: float = 0.8,
        im_folder: str = "images",
        annotation_folder: str = "annotations",
    ):
        """ initialize dataset

        This class assumes that the data is formatted in two folders:
            - annotation folder which contains the Pascal VOC formatted
              annotations
            - image folder which contains the images

        Args:
            root: the root path of the dataset containing the image and
            annotation folders
            batch_size: batch size for dataloaders
            transforms: the transformations to apply
            train_test_ratio: the ratio of training to testing data
            im_folder: the name of the image folder
            annotation_folder: the name of the annotation folder
        """

        self.root = Path(root)
        self.transforms = transforms
        self.im_folder = im_folder
        self.anno_folder = annotation_folder
        self.batch_size = batch_size
        self.train_test_ratio = train_test_ratio

        # get images and annotations
        self.im_paths = list(sorted(os.listdir(self.root / self.im_folder)))
        self.anno_paths = list(
            sorted(os.listdir(self.root / self.anno_folder))
        )
        self.categories = self._get_categories()

        # setup datasets (train_ds, test_ds, train_dl, test_dl)
        self._setup_data(train_test_ratio=train_test_ratio)

    def _get_categories(self) -> List[str]:
        """ Parses all Pascal VOC formatted annotation files to extract all
        possible categories. """
        categories = ["__background__"]
        for anno_path in self.anno_paths:
            anno_path = self.root / "annotations" / str(anno_path)
            tree = ET.parse(anno_path)
            root = tree.getroot()
            objs = root.findall("object")
            for obj in objs:
                category = obj[0]
                assert category.tag == "name"
                categories.append(category.text)
        return list(set(categories))

    def split_train_test(
        self, train_test_ratio: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set

        Args:
            train_test_ratio: the ratio of images to use for training vs
            testing

        Return
            A training and testing dataset in that order
        """
        test_num = math.floor(len(self) * (1 - train_test_ratio))
        indices = torch.randperm(len(self)).tolist()
        self.transforms = get_transform(train=True)
        train = Subset(self, indices[:-test_num])
        self.transforms = get_transform(train=False)
        test = Subset(self, indices[-test_num:])
        return train, test

    def _setup_data(self, train_test_ratio: float = 0.8):
        """ create training and validation data loaders
        """
        self.train_ds, self.test_ds = self.split_train_test(
            train_test_ratio=train_test_ratio
        )

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def show_ims(self, rows: int = 1, rand: bool = False) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display
            rand: randomize images

        Returns None but displays a grid of annotated images.
        """
        im_features = partial(self._read_anno_idx, None, True)
        plot_grid(display_bounding_boxes, im_features, rows=rows)

    def _read_anno_path(
        self, anno_path: str
    ) -> Tuple[List[Annotation], Union[str, Path]]:
        """ Extract the annotations and image path from labelling in Pascal VOC format.

        Args:
            anno_path: the path to the annotation xml file

        Return
            A tuple of annotations and the image path
        """
        annos = []
        tree = ET.parse(anno_path)
        root = tree.getroot()

        # extract bounding boxes and classification
        objs = root.findall("object")
        for obj in objs:
            category = obj[0]
            assert category.tag == "name"

            bnd_box = obj[4]
            assert bnd_box.tag == "bndbox"

            left = int(bnd_box[0].text)
            top = int(bnd_box[1].text)
            right = int(bnd_box[2].text)
            bottom = int(bnd_box[3].text)

            bbox = Bbox(left, top, right, bottom)
            assert bbox.is_valid()

            anno = Annotation(
                bbox=bbox,
                category_name=category.text,
                category_idx=self.categories.index(category.text),
            )
            annos.append(anno)

        # get image path from annotation
        anno_dir = os.path.dirname(anno_path)
        im_path = os.path.realpath(
            os.path.join(anno_dir, root.find("path").text)
        )

        return (annos, im_path)

    def _read_anno_idx(
        self, idx: int, rand: bool = False
    ) -> Tuple[List[Annotation], Union[str, Path]]:
        """ Get annotation by index

        Args:
            idx: the index to read from
            rand: choose random index

        Raises:
            Exception if idx is not None and rand is set to True
            Exception if rand and idx is set to False and None respectively

        Returns a list of annotaitons and the image path
        """

        if (idx is not None) and (rand is True):
            raise Exception("idx cannot be set if rand is set to True.")
        if idx is None and rand is False:
            raise Exception("specify idx if rand is False.")

        if rand:
            idx = randrange(len(self.im_paths))

        anno_path = self.root / self.anno_folder / str(self.anno_paths[idx])
        return self._read_anno_path(anno_path)

    def __getitem__(self, idx):
        """ Make iterable. """
        # get box/labels from annotations
        annos, im_path = self._read_anno_idx(idx)

        boxes = [
            [anno.bbox.left, anno.bbox.top, anno.bbox.right, anno.bbox.bottom]
            for anno in annos
        ]
        labels = [anno.category_idx for anno in annos]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # get area for evaluation with the COCO metric, to separate the
        # metric scores between small, medium and large boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd (torchvision specific)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # unique id
        im_id = torch.tensor([idx])

        # setup target dic
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": im_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # get image
        im = Image.open(im_path).convert("RGB")

        # and apply transforms if any
        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return (im, target)

    def __len__(self):
        return len(self.im_paths)
