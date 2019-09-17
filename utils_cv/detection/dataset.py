# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import math
from pathlib import Path
from random import randrange
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image

from .plot import display_bboxes, plot_grid
from .bbox import AnnotationBbox
from .references.utils import collate_fn
from .references.transforms import RandomHorizontalFlip, Compose, ToTensor
from utils_cv.common.gpu import db_num_workers


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
        # TODO we can add more 'default' transformations here
    return Compose(transforms)


def parse_pascal_voc_anno(
    anno_path: str, labels: List[str] = None
) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
    """ Extract the annotations and image path from labelling in Pascal VOC format.

    Args:
        anno_path: the path to the annotation xml file
        labels: list of all possible labels, used to compute label index for each label name

    Return
        A tuple of annotations and the image path
    """

    anno_bboxes = []
    tree = ET.parse(anno_path)
    root = tree.getroot()

    # get image path from annotation. Note that the path field might not be set.
    anno_dir = os.path.dirname(anno_path)
    if root.find("path"):
        im_path = os.path.realpath(
            os.path.join(anno_dir, root.find("path").text)
        )
    else:
        im_path = os.path.realpath(
            os.path.join(anno_dir, root.find("filename").text)
        )

    # extract bounding boxes and classification
    objs = root.findall("object")
    for obj in objs:
        label = obj.find("name").text
        bnd_box = obj.find("bndbox")
        left = int(bnd_box[0].text)
        top = int(bnd_box[1].text)
        right = int(bnd_box[2].text)
        bottom = int(bnd_box[3].text)

        # Set mapping of label name to label index
        if labels is None:
            label_idx = None
        else:
            label_idx = labels.index(label)

        anno_bbox = AnnotationBbox.from_array(
            [left, top, right, bottom],
            label_name=label,
            label_idx=label_idx,
            im_path=im_path,
        )
        assert anno_bbox.is_valid()
        anno_bboxes.append(anno_bbox)

    return (anno_bboxes, im_path)


class DetectionDataset:
    """ An object detection dataset.

    The dunder methods __init__, __getitem__, and __len__ were inspired from code found here:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan
    """

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 2,
        transforms: object = get_transform(train=True),
        train_pct: float = 0.5,
        anno_dir: str = "annotations",
        im_dir: str = "images",
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
            train_pct: the ratio of training to testing data
            annotation_dir: the name of the annotation subfolder under the root directory
            im_dir: the name of the image subfolder under the root directory. If set to 'None' then infers image location from annotation .xml files
        """

        self.root = Path(root)
        # TODO think about how transforms are working...
        self.transforms = transforms
        self.im_dir = im_dir
        self.anno_dir = anno_dir
        self.batch_size = batch_size
        self.train_pct = train_pct

        # read annotations
        self._read_annos()

        # create training and validation datasets
        self.train_ds, self.test_ds = self.split_train_test(
            train_pct=train_pct
        )

        # create training and validation data loaders
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=db_num_workers(),
            collate_fn=collate_fn,
        )

        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=db_num_workers(),
            collate_fn=collate_fn,
        )

    def _read_annos(self) -> List[str]:
        """ Parses all Pascal VOC formatted annotation files to extract all
        possible labels. """

        # All annotation files are assumed to be in the anno_dir directory.
        # If im_dir is provided then find all images in that directory, and
        # it's assumed that the annotation filenames end with .xml.
        # If im_dir is not provided, then the image paths are read from inside
        # the .xml annotations.
        if self.im_dir is None:
            anno_filenames = sorted(os.listdir(self.root / self.anno_dir))
            self.im_paths = []
        else:
            im_filenames = sorted(os.listdir(self.root / self.im_dir))
            anno_filenames = [
                os.path.splitext(s)[0] + ".xml" for s in im_filenames
            ]
            self.im_paths = [
                os.path.join(self.root / self.im_dir, s) for s in im_filenames
            ]

        # Parse all annotations
        self.anno_paths = []
        self.anno_bboxes = []
        for anno_filename in anno_filenames:
            anno_path = self.root / self.anno_dir / str(anno_filename)
            assert os.path.exists(anno_path), (
                f"Cannot find annotation file: {anno_path}"
            )
            anno_bboxes, im_path = parse_pascal_voc_anno(anno_path)

            # TODO For now, ignore all images without a single bounding box in it, otherwise throws error during training.
            if len(anno_bboxes) == 0:
                continue

            if self.im_dir is None:
                self.im_paths.append(im_path)
            self.anno_paths.append(anno_path)
            self.anno_bboxes.append(anno_bboxes)

        # Get list of all labels
        labels = []
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                labels.append(anno_bbox.label_name)
        self.labels = list(set(labels))

        # Set for each bounding box label name also what its integer representation is
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                anno_bbox.label_idx = self.labels.index(anno_bbox.label_name) +1   

    def split_train_test(
        self, train_pct: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set

        Args:
            train_pct: the ratio of images to use for training vs
            testing

        Return
            A training and testing dataset in that order
        """
        # TODO Is it possible to make these lines in split_train_test() a bit
        # more intuitive?

        test_num = math.floor(len(self) * (1 - train_pct))
        indices = torch.randperm(len(self)).tolist()

        self.transforms = get_transform(train=True)
        train = Subset(self, indices[test_num:])

        self.transforms = get_transform(train=False)
        test = Subset(self, indices[: test_num + 1])

        return train, test

    def show_ims(self, rows: int = 1, cols: int = 3) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display
            cols: cols to display, NOTE: use 3 for best looking grid

        Returns None but displays a grid of annotated images.
        """
        plot_grid(display_bboxes, self._get_random_anno, rows=rows, cols=cols)

    def _get_random_anno(
        self
    ) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
        """ Get random annotation and corresponding image

        Returns a list of annotatoons and the image path
        """
        idx = randrange(len(self.anno_paths))
        return self.anno_bboxes[idx], self.im_paths[idx]

    def __getitem__(self, idx):
        """ Make iterable. """
        # get box/labels from annotations
        im_path = self.im_paths[idx]
        anno_bboxes = self.anno_bboxes[idx]
        boxes = [
            [anno_bbox.left, anno_bbox.top, anno_bbox.right, anno_bbox.bottom]
            for anno_bbox in anno_bboxes
        ]
        labels = [anno_bbox.label_idx for anno_bbox in anno_bboxes]

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
        return len(self.anno_paths)
