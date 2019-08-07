# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Union
from pathlib import Path
from random import randrange
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from PIL import Image
from .plot import display_bounding_boxes
from .model import get_transform


class DetectionDataset(object):
    """ An object detection dataset.
    """
    def __init__(
        self,
        root: Union[str, Path],
        # categories: List[str],
        transforms: object = None,
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
        """

        self.root = Path(root)
        self.transforms = transforms

        self.ims = list(sorted(os.listdir(self.root / "images")))
        self.annotations = list(sorted(os.listdir(self.root / "annotations")))
        self.categories = self._get_categories()

    def _get_categories(self) -> List[str]:
        """ Parses all Pascal VOC formatted annoatation files to extract all
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

    def get_random_image(self) -> Tuple[List[List[int]], List[str], str]:
        """ Choose and get image from dataset.

        This function returns all the data that is needed to effectively
        visualize an image from the dataset.

        Returns:
            A tuple of boxes, categories, image path
        """
        rand_idx = randrange(len(self.ims))
        boxes, labels, im_path = self._get_im_data(rand_idx)
        return (boxes, [self.categories[label] for label in labels], im_path)

    def split_train_test(
        self, train_ratio: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split theis dataset into a training and testing set

        Args:
            train_ratio: the amount of images to use for training (the rest
            will be used for testing.
        Return
            A training and testing dataset in that order
        """
        indices = torch.randperm(len(self)).tolist()
        self.set_transform(get_transform(train=True))
        train = Subset(self, indices[:-50])
        self.set_transform(get_transform(train=False))
        test = Subset(self, indices[-50:])
        return train, test

    def set_transform(self, transforms: List[object]) -> None:
        """ Apply transformations. """
        self.transforms = transforms

    def show_batch(
        self, rows: int = 1, figsize: Tuple[int, int] = (16, 16),
    ) -> None:
        """ Show batch of images.

        Args:
            rows: the number of rows images to display
            figize: the figure size to use

        Returns None but displays a grid of annotated images.
        """
        fig, axes = plt.subplots(rows, 3, figsize=figsize)
        for row in axes:
            for ax in row:
                display_bounding_boxes(*self.get_random_image(), ax)
        plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.1, wspace=0.2)

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
            self.root / "annotations" / str(self.annotations[idx])
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
