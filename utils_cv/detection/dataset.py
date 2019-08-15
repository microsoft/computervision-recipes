# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import math
from pathlib import Path
from random import randrange
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset
import xml.etree.ElementTree as ET
from PIL import Image

from .plot import display_bounding_boxes, plot_grid
from .model import get_transform


class DetectionDataset(object):
    """ An object detection dataset.

    The dunder methods __init__, __getitem__, and __len__ were inspired from code found here:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan
    """

    def __init__(
        self,
        root: Union[str, Path],
        transforms: object = None,
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
            transforms: the transformations to apply
            image_folder: the name of the image folder
            annotation_folder: the name of the annotation folder
        """

        self.root = Path(root)
        self.transforms = transforms
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder

        self.ims = list(sorted(os.listdir(self.root / self.image_folder)))
        self.annotations = list(
            sorted(os.listdir(self.root / self.annotation_folder))
        )
        self.categories = self._get_categories()

    def _get_categories(self) -> List[str]:
        """ Parses all Pascal VOC formatted annotation files to extract all
        possible categories. """
        categories = ["__background__"]
        for annotation_path in self.annotations:
            annotation_path = self.root / "annotations" / str(annotation_path)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            objs = root.findall("object")
            for obj in objs:
                category = obj[0]
                assert category.tag == "name"
                categories.append(category.text)
        return list(set(categories))

    def get_image_features(
        self, idx: int = None, rand: bool = True
    ) -> Tuple[List[List[int]], List[str], str]:
        """ Choose and get image from dataset.

        This function returns all the data that is needed to effectively
        visualize an image from the dataset.

        Args:
            idx: The image index to get the features of
            rand: randomly select image (default is true)

        Raises:
            Exception if idx is not None and rand is set to True
            Exception if rand and idx is set to False and None respectively

        Returns:
            A tuple of boxes, categories, image path
        """

        if (idx is not None) and (rand is True):
            raise Exception("idx cannot be set if rand is set to True.")
        if idx is None and rand is False:
            raise Exception(
                "specify idx if rand is True (which is the default setting)."
            )

        if rand:
            idx = randrange(len(self.ims))

        boxes, labels, im_path = self._get_im_data(idx)
        return (boxes, [self.categories[label] for label in labels], im_path)

    def split_train_test(
        self, train_ratio: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set

        Args:
            train_ratio: the ratio of images to use for training (the rest
            will be used for testing.

        Return
            A training and testing dataset in that order
        """
        test_num = math.floor(len(self) * (1 - train_ratio))
        indices = torch.randperm(len(self)).tolist()
        self.transforms = get_transform(train=True)
        train = Subset(self, indices[:-test_num])
        self.transforms = get_transform(train=False)
        test = Subset(self, indices[-test_num:])
        return train, test

    def show_batch(self, rows: int = 1) -> None:
        """ Show batch of images.

        Args:
            rows: the number of rows images to display

        Returns None but displays a grid of annotated images.
        """
        plot_grid(display_bounding_boxes, self.get_image_features, rows=rows)

    def _get_annotations(
        self, annotation_path: str
    ) -> Tuple[List[List[str]], List[int], str]:
        """ Extract the annotations and image path from labelling in Pascal VOC format.

        Args:
            annotation_path: the path to the annotation xml file

        Return
            A tuple of boxes, labels, and the image path
        """
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
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

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.categories.index(category.text))

        # get image path from annotation
        annotation_dir = os.path.dirname(annotation_path)
        im_path = root.find("path").text
        im_path = os.path.realpath(os.path.join(annotation_dir, im_path))

        return (boxes, labels, im_path)

    def _get_im_data(self, idx) -> Tuple[List[List[int]], List[int], str]:
        """
        Returns
            (boxes, labels, im_path)
        """
        annotation_path = (
            self.root / self.annotation_folder / str(self.annotations[idx])
        )
        return self._get_annotations(annotation_path)

    def __getitem__(self, idx):
        """ Make iterable. """
        # get box/labels from annotations
        boxes, labels, im_path = self._get_im_data(idx)

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
        return len(self.ims)
