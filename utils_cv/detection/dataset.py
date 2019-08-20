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


class Bbox:
    """ Util to represent bounding boxes

    Source:
    https://github.com/Azure/ObjectDetectionUsingCntk/blob/master/helpers.py
    """

    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = int(round(float(left)))
        self.top = int(round(float(top)))
        self.right = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return "Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}".format(
            self.left, self.top, self.right, self.bottom
        )

    def __repr__(self):
        return str(self)

    def rect(self) -> List[int]:
        return [self.left, self.top, self.right, self.bottom]

    def max(self) -> int:
        return max([self.left, self.top, self.right, self.bottom])

    def min(self) -> int:
        return min([self.left, self.top, self.right, self.bottom])

    def width(self) -> int:
        width = self.right - self.left + 1
        assert width >= 0
        return width

    def height(self) -> int:
        height = self.bottom - self.top + 1
        assert height >= 0
        return height

    def surface_area(self) -> float:
        return self.width() * self.height()

    def get_overlap_bbox(self, bbox: "Bbox") -> Union[None, "Bbox"]:
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlap_left = max(left1, left2)
        overlap_top = max(top1, top2)
        overlap_right = min(right1, right2)
        overlap_bottom = min(bottom1, bottom2)
        if (overlap_left > overlap_right) or (overlap_top > overlap_bottom):
            return None
        else:
            return Bbox(
                overlap_left, overlap_top, overlap_right, overlap_bottom
            )

    def standardize(
        self
    ) -> None:  # NOTE: every setter method should call standardize
        left_new = min(self.left, self.right)
        top_new = min(self.top, self.bottom)
        right_new = max(self.left, self.right)
        bottom_new = max(self.top, self.bottom)
        self.left = left_new
        self.top = top_new
        self.right = right_new
        self.bottom = bottom_new

    def crop(self, max_width: int, max_height: int) -> "Bbox":
        left_new = min(max(self.left, 0), maxWidth)
        top_new = min(max(self.top, 0), maxHeight)
        right_new = min(max(self.right, 0), maxWidth)
        bottom_new = min(max(self.bottom, 0), maxHeight)
        return Bbox(left_new, top_new, right_new, bottom_new)

    def is_valid(self) -> bool:
        if self.left >= self.right or self.top >= self.bottom:
            return False
        if (
            min(self.rect()) < -self.MAX_VALID_DIM
            or max(self.rect()) > self.MAX_VALID_DIM
        ):
            return False
        return True


class Annotation:
    """ Contains a Bbox. """

    def __init__(self, bbox: Bbox, category_name: str, category_idx: int):
        self.bbox = bbox
        self.category_name = category_name
        self.category_idx = category_idx


class DetectionDataset(object):
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
        image_folder: str = "images",
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
            image_folder: the name of the image folder
            annotation_folder: the name of the annotation folder
        """

        self.root = Path(root)
        self.transforms = transforms
        self.image_folder = image_folder
        self.anno_folder = annotation_folder
        self.batch_size = batch_size
        self.train_test_ratio = train_test_ratio

        # get images and annotations
        self.im_paths = list(sorted(os.listdir(self.root / self.image_folder)))
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

    def get_image_features(
        self, idx: int = None, rand: bool = False
    ) -> Tuple[List[List[int]], List[str], str]:
        """ Choose and get image from dataset.

        This function returns all the data that is needed to effectively
        visualize an image from the dataset.

        Args:
            idx: The image index to get the features of
            rand: randomly select image

        Raises:
            Exception if idx is not None and rand is set to True
            Exception if rand and idx is set to False and None respectively

        Returns:
            A tuple of boxes, categories, image path
        """

        if (idx is not None) and (rand is True):
            raise Exception("idx cannot be set if rand is set to True.")
        if idx is None and rand is False:
            raise Exception("specify idx if rand is True.")

        if rand:
            idx = randrange(len(self.im_paths))

        boxes, labels, im_path = self._read_anno_idx(idx)
        return (boxes, [self.categories[label] for label in labels], im_path)

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

    def show_ims(self, rows: int = 1) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display

        Returns None but displays a grid of annotated images.
        """
        get_image_features = partial(self.get_image_features, rand=True)
        plot_grid(display_bounding_boxes, get_image_features, rows=rows)

    def _read_anno_path(
        self, anno_path: str
    ) -> Tuple[List[List[str]], List[int], str]:
        """ Extract the annotations and image path from labelling in Pascal VOC format.

        Args:
            anno_path: the path to the annotation xml file

        Return
            A tuple of boxes, labels, and the image path
        """
        boxes = []
        labels = []
        # annos = []
        tree = ET.parse(anno_path)
        root = tree.getroot()

        # extract bounding boxes and classification
        objs = root.findall("object")
        for obj in objs:
            category = obj[0]
            assert category.tag == "name"

            bnd_box = obj[4]
            assert bnd_box.tag == "bndbox"

            xmin = int(bnd_box[0].text)
            ymin = int(bnd_box[1].text)
            xmax = int(bnd_box[2].text)
            ymax = int(bnd_box[3].text)

            # bbox = Bbox(xmin, ymin, xmax, ymax)
            # assert bbox.is_valid()

            # anno = Annotation(
            #     bbox=bbox,
            #     category_name=category.text,
            #     category_idx=self.categories.index(category.text)
            # )
            # annos.append(anno)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.categories.index(category.text))

        # get image path from annotation
        anno_dir = os.path.dirname(anno_path)
        im_path = os.path.realpath(
            os.path.join(anno_dir, root.find("path").text)
        )

        # return annos, im_path
        return (boxes, labels, im_path)

    def _read_anno_idx(self, idx) -> Tuple[List[List[int]], List[int], str]:
        """
        Returns
            (boxes, labels, im_path)
        """
        anno_path = self.root / self.anno_folder / str(self.anno_paths[idx])
        return self._read_anno_path(anno_path)

    def __getitem__(self, idx):
        """ Make iterable. """
        # get box/labels from annotations
        boxes, labels, im_path = self._read_anno_idx(idx)

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
