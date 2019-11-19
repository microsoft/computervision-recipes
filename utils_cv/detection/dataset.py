# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import copy
import math
import numpy as np
from pathlib import Path
import random
from typing import Callable, List, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import ColorJitter
import xml.etree.ElementTree as ET
from PIL import Image

from .plot import display_annotations, plot_grid
from .bbox import AnnotationBbox
from .mask import binarise_mask
from .references.utils import collate_fn
from .references.transforms import RandomHorizontalFlip, Compose, ToTensor
from ..common.gpu import db_num_workers

Trans = Callable[[object, dict], Tuple[object, dict]]


class ColorJitterTransform(object):
    """ Wrapper for torchvision's ColorJitter to make sure 'target
    object is passed along """

    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, im, target):
        im = ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )(im)
        return im, target


def get_transform(train: bool) -> Trans:
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

    # transformations to apply before image is turned into a tensor
    if train:
        transforms.append(
            ColorJitterTransform(
                brightness=0.2, contrast=0.2, saturation=0.4, hue=0.05
            )
        )

    # transform im to tensor
    transforms.append(ToTensor())

    # transformations to apply after image is turned into a tensor
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

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
    if root.find("path") is not None:
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
        left = int(bnd_box.find("xmin").text)
        top = int(bnd_box.find("ymin").text)
        right = int(bnd_box.find("xmax").text)
        bottom = int(bnd_box.find("ymax").text)

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

    return anno_bboxes, im_path


class DetectionDataset:
    """ An object detection dataset.

    The implementation of the dunder methods __init__, __getitem__, and __len__ were inspired from code found here:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan
    """

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 2,
        train_transforms: Trans = get_transform(train=True),
        test_transforms: Trans = get_transform(train=False),
        train_pct: float = 0.5,
        anno_dir: str = "annotations",
        im_dir: str = "images",
        mask_dir: str = None,
        seed: int = None,
        allow_negatives: bool = False
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
            train_transforms: the transformations to apply to the train set
            test_transforms: the transformations to apply to the test set
            train_pct: the ratio of training to testing data
            anno_dir: the name of the annotation subfolder under the root directory
            im_dir: the name of the image subfolder under the root directory. If set to 'None' then infers image location from annotation .xml files
            allow_negatives: is false (default) then will throw an error if no anntation .xml file can be found for a given image. Otherwise use image as negative, ie assume that the image does not contain any of the objects of interest.
            mask_dir: the name of the mask subfolder under the root directory if the dataset is used for instance segmentation
            seed: random seed for splitting dataset to training and testing data
        """

        self.root = Path(root)
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.im_dir = im_dir
        self.anno_dir = anno_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.allow_negatives = allow_negatives
        self.seed = seed

        # read annotations
        self._read_annos()

        # create training and validation datasets
        self.train_ds, self.test_ds = self.split_train_test(
            train_pct=train_pct
        )

        # create training and validation data loaders
        self.init_data_loaders()

    def _read_annos(self) -> None:
        """ Parses all Pascal VOC formatted annotation files to extract all
        possible labels. """

        # All annotation files are assumed to be in the anno_dir directory.
        # If im_dir is provided then find all images in that directory, and
        # it's assumed that the annotation filenames end with .xml.
        # If im_dir is not provided, then the image paths are read from inside
        # the .xml annotations.
        if self.im_dir is None:
            anno_filenames = sorted(os.listdir(self.root / self.anno_dir))
        else:
            im_filenames = sorted(os.listdir(self.root / self.im_dir))
            im_paths = [
                os.path.join(self.root / self.im_dir, s) for s in im_filenames
            ]
            anno_filenames = [
                os.path.splitext(s)[0] + ".xml" for s in im_filenames
            ]

        # Read all annotations
        self.im_paths = []
        self.anno_paths = []
        self.anno_bboxes = []
        self.mask_paths = []
        for anno_idx, anno_filename in enumerate(anno_filenames):
            anno_path = self.root / self.anno_dir / str(anno_filename)

            # Parse annotation file if present
            if os.path.exists(anno_path):
                anno_bboxes, im_path = parse_pascal_voc_anno(anno_path)
            else:
                if not self.allow_negatives:
                    raise FileNotFoundError(anno_path)
                anno_bboxes = []
                im_path = im_paths[anno_idx]

            # Torchvision needs at least one ground truth bounding box per image. Hence for images without a single
            # annotated object, adding a tiny bounding box with "background" label 0.
            if len(anno_bboxes) == 0:
                anno_bboxes = [
                    AnnotationBbox.from_array(
                        [1, 1, 5, 5],
                        label_name=None,
                        label_idx=0,
                        im_path=im_path,
                    )
                ]

            if self.im_dir is None:
                self.im_paths.append(im_path)
            else:
                self.im_paths.append(im_paths[anno_idx])

            if self.mask_dir:
                # Assume mask image name matches image name but has .png
                # extension
                mask_name = os.path.basename(self.im_paths[-1])
                mask_name = mask_name[:mask_name.rindex('.')] + ".png"
                mask_path = self.root / self.mask_dir / mask_name
                # For mask prediction, if no mask provided and negatives not
                # allowed (), ignore the image
                if not mask_path.exists():
                    if not self.allow_negatives:
                        raise FileNotFoundError(mask_path)
                    else:
                        self.mask_paths.append(None)
                else:
                    self.mask_paths.append(mask_path)

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
                if (
                    anno_bbox.label_name is None
                ):  # background rectangle is assigned id 0 by design
                    anno_bbox.label_idx = 0
                else:
                    anno_bbox.label_idx = (
                        self.labels.index(anno_bbox.label_name) + 1
                    )

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
        test_num = math.floor(len(self) * (1 - train_pct))
        if self.seed:
            torch.manual_seed(self.seed)
        indices = torch.randperm(len(self)).tolist()

        train = copy.deepcopy(Subset(self, indices[test_num:]))
        train.dataset.transforms = self.train_transforms

        test = copy.deepcopy(Subset(self, indices[:test_num]))
        test.dataset.transforms = self.test_transforms

        return train, test

    def init_data_loaders(self):
        """ Create training and validation data loaders """
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

    def add_images(
        self,
        im_paths: List[str],
        anno_bboxes: List[AnnotationBbox],
        target: str = "train",
        mask_paths: List[str] = None,
    ):
        """ Add new images to either the training or test set.

        Args:
            im_paths: path to the images.
            anno_bboxes: ground truth boxes for each image.
            target: specify if images are to be added to the training or test set. Valid options: "train" or "test".
            mask_paths: path to the masks.

        Raises:
            Exception if `target` variable is neither 'train' nor 'test'
        """
        assert len(im_paths) == len(anno_bboxes)
        for i, (im_path, anno_bbox) in enumerate(zip(im_paths, anno_bboxes)):
            self.im_paths.append(im_path)
            self.anno_bboxes.append(anno_bbox)
            if mask_paths is not None:
                self.mask_paths.append(mask_paths[i])
            if target.lower() == "train":
                self.train_ds.dataset.im_paths.append(im_path)
                self.train_ds.dataset.anno_bboxes.append(anno_bbox)
                if mask_paths is not None:
                    self.train_ds.dataset.mask_paths.append(mask_paths[i])
                self.train_ds.indices.append(len(self.im_paths) - 1)
            elif target.lower() == "test":
                self.test_ds.dataset.im_paths.append(im_path)
                self.test_ds.dataset.anno_bboxes.append(anno_bbox)
                if mask_paths is not None:
                    self.test_ds.dataset.mask_paths.append(mask_paths[i])
                self.test_ds.indices.append(len(self.im_paths) - 1)
            else:
                raise Exception(f"Target {target} unknown.")

        # Re-initialize the data loaders
        self.init_data_loaders()

    def show_ims(self, rows: int = 1, cols: int = 3, seed: int = None) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display
            cols: cols to display, NOTE: use 3 for best looking grid
            seed: random seed for selecting images

        Returns None but displays a grid of annotated images.
        """
        if seed or self.seed:
            random.seed(seed or self.seed)

        plot_grid(display_annotations, self._get_random_anno, rows=rows, cols=cols)

    def show_im_transformations(
        self, idx: int = None, rows: int = 1, cols: int = 3
    ) -> None:
        """ Show a set of images after transformations have been applied.

        Args:
            idx: the index to of the image to show the transformations for.
            rows: number of rows to display
            cols: number of cols to display, NOTE: use 3 for best looking grid

        Returns None but displays a grid of randomly applied transformations.
        """
        if not hasattr(self, "transforms"):
            print(
                (
                    "Transformations are not applied ot the base dataset object.\n"
                    "Call this function on either the train_ds or test_ds instead:\n\n"
                    "    my_detection_data.train_ds.dataset.show_im_transformations()"
                )
            )
        else:
            if idx is None:
                idx = random.randrange(len(self.anno_paths))

            def plotter(im, ax):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(im)

            def im_gen() -> torch.Tensor:
                return self[idx][0].permute(1, 2, 0)

            plot_grid(plotter, im_gen, rows=rows, cols=cols)

            print(f"Transformations applied on {self.im_paths[idx]}:")
            [print(transform) for transform in self.transforms.transforms]

    def _get_binary_mask(self, idx: int) -> Union[np.ndarray, None]:
        """ Return binary masks for objects in the mask image. """
        binary_masks = None
        if self.mask_paths:
            if self.mask_paths[idx] is not None:
                binary_masks = binarise_mask(Image.open(self.mask_paths[idx]))
            else:
                # for the tiny bounding box in _read_annos(), make the mask to
                # be the whole box
                mask = np.zeros(
                    Image.open(self.im_paths[idx]).size[::-1],
                    dtype=np.uint8
                )
                binary_masks = binarise_mask(mask)

        return binary_masks

    def _get_random_anno(self) -> Tuple:
        """ Get random annotation and corresponding image

        Returns a list of annotations and the image path
        """
        idx = random.randrange(len(self.im_paths))
        return self.anno_bboxes[idx], self.im_paths[idx], self._get_binary_mask(idx)

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

        # get masks
        binary_masks = self._get_binary_mask(idx)
        if binary_masks is not None:
            target["masks"] = torch.as_tensor(binary_masks, dtype=torch.uint8)

        # get image
        im = Image.open(im_path).convert("RGB")

        # and apply transforms if any
        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def __len__(self):
        return len(self.im_paths)
