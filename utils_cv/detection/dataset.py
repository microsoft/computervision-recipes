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
        annotation_dir: str = "annotations",
        im_dir: str = None,
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
        self.anno_dir = annotation_dir
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

        # Parse all annotations
        self.im_paths = []
        self.anno_paths = []
        self.anno_bboxes = []
        anno_filenames = sorted(os.listdir(self.root / self.anno_dir))

        for anno_filename in anno_filenames:
            anno_path = self.root / self.anno_dir / str(anno_filename)
            anno_bboxes, im_path = self._parse_anno(anno_path)

            # TODO For now, ignore all images without a single bounding box in it, otherwise throws error during training.
            if len(anno_bboxes) == 0:
                continue

            self.im_paths.append(im_path)
            self.anno_paths.append(anno_path)
            self.anno_bboxes.append(anno_bboxes)

        # Get list of all labels
        labels = ["__background__"]
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                labels.append(anno_bbox.label_name)
        self.labels = list(set(labels))

        # Set for each bounding box label name also what its integer representation is
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                anno_bbox.label_idx = self.labels.index(anno_bbox.label_name)

    def _parse_anno(
        self, anno_path: str
    ) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
        """ Extract the annotations and image path from labelling in Pascal VOC format.

            Args:
                anno_path: the path to the annotation xml file

            Return
                A tuple of annotations and the image path
            """

        anno_bboxes = []
        tree = ET.parse(anno_path)
        root = tree.getroot()

        # get image path from annotation
        anno_dir = os.path.dirname(anno_path)
        if self.im_dir is None:
            im_path = os.path.realpath(
                os.path.join(anno_dir, root.find("path").text)
            )
        else:
            anno_name_without_extension = os.path.splitext(
                os.path.basename(anno_path)
            )[0]
            im_filename = anno_name_without_extension + ".jpg"
            im_path = os.path.join(self.root / self.im_dir, im_filename)
        assert os.path.exists(im_path), "Image does not exist: " + str(im_path)

        # extract bounding boxes and classification
        objs = root.findall("object")
        for obj in objs:
            label = obj.find("name")
            bnd_box = obj.find("bndbox")
            left = int(bnd_box[0].text)
            top = int(bnd_box[1].text)
            right = int(bnd_box[2].text)
            bottom = int(bnd_box[3].text)

            anno_bbox = AnnotationBbox.from_array(
                [left, top, right, bottom],
                label_name=label.text,
                label_idx=None,  # set mapping from label name to label index later once list of all labels is known
                im_path=im_path,
            )
            assert anno_bbox.is_valid()

            anno_bboxes.append(anno_bbox)

        return (anno_bboxes, im_path)

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

    def show_ims(
        self, rows: int = 1, cols: int = 3, rand: bool = False
    ) -> None:
        """ Show a set of images.

        Args:
            rows: the number of rows images to display
            cols: cols to display, NOTE: use 3 for best looking grid
            rand: randomize images

        Returns None but displays a grid of annotated images.
        """
        im_annos = partial(self._get_random_anno, None, True)
        plot_grid(display_bboxes, im_annos, rows=rows, cols=cols)

    def _get_random_anno(
        self
    ) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
        """ Get random annotation and corresponding image

        Returns a list of annotatoons and the image path
        """

        idx = randrange(len(self.anno_paths))
        return self.anno_bboxes[idx], self.im_paths[idx]

    # def _get_anno_idx(
    #     self, idx: int, rand: bool = False
    # ) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
    #     """ Get annotation by index

    #     Args:
    #         idx: the index to read from
    #         rand: choose random index

    #     Raises:
    #         Exception if idx is not None and rand is set to True
    #         Exception if rand and idx is set to False and None respectively

    #     Returns a list of annotaitons and the image path
    #     """

    #     if (idx is not None) and (rand is True):
    #         raise Exception("idx cannot be set if rand is set to True.")
    #     if idx is None and rand is False:
    #         raise Exception("specify idx if rand is False.")

    #     if rand:
    #         idx = randrange(len(self.anno_paths))

    #     return self.anno_bboxes[idx], self.im_paths[idx]

    # def get_path_from_idx(self, idx: int) -> str:
    #     """ Gets an im_path from idx. """
    #     #return self.root / self.anno_dir / str(self.anno_paths[idx])

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
