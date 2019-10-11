# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import math
import numpy as np
from pathlib import Path
import random
from typing import Callable, List, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image

from .plot import display_bboxes, display_bbox_mask, display_bbox_mask_keypoint, plot_grid
from .bbox import AnnotationBbox
from .data import coco_labels, Urls
from .mask import binarise_mask
from .references.utils import collate_fn
from .references.transforms import Compose, RandomHorizontalFlip, ToTensor
from ..common.data import data_path, get_files_in_directory, unzip_url
from ..common.gpu import db_num_workers

Trans = Callable[[object, dict], Tuple[object, dict]]


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
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        # TODO we can add more 'default' transformations here
    return Compose(transforms)


def parse_pascal_voc_anno(
    anno_path: str, labels: List[str] = None
) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
    """ Extract the annotations and image path from labelling in Pascal VOC
    format.

    Args:
        anno_path: the path to the annotation xml file
        labels: list of all possible labels, used to compute label index for
                each label name

    Return
        A tuple of annotations and the image path
    """

    anno_bboxes = []
    tree = ET.parse(anno_path)
    root = tree.getroot()

    # get image path from annotation. Note that the path field might not be
    # set.
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
        left = int(bnd_box.find('xmin').text)
        top = int(bnd_box.find('ymin').text)
        right = int(bnd_box.find('xmax').text)
        bottom = int(bnd_box.find('ymax').text)

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


class DetectionDataset(Dataset):
    """ An object detection dataset.

    The dunder methods __init__, __getitem__, and __len__ were inspired from
    code found here:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan
    """

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 2,
        transforms: Union[Trans, Tuple[Trans, Trans]] = (
                get_transform(train=True),
                get_transform(train=False)
        ),
        train_pct: float = 0.5,
        anno_dir: str = "annotations",
        im_dir: str = "images",
        seed: int = None,
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
            anno_dir: the name of the annotation subfolder under the root
                      directory
            im_dir: the name of the image subfolder under the root directory.
                    If set to 'None' then infers image location from annotation
                    .xml files
        """

        self.root = Path(root)
        # TODO think about how transforms are working...
        if transforms and len(transforms) == 1:
            self.transforms = (transforms, ) * 2
        self.transforms = transforms
        self.im_dir = im_dir
        self.anno_dir = anno_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.seed = seed

        # read annotations
        self._read_annos()

        self._get_dataloader(train_pct)

    def _get_dataloader(self, train_pct):
        # create training and validation datasets
        train_ds, test_ds = self.split_train_test(
            train_pct=train_pct
        )

        # create training and validation data loaders
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=db_num_workers(),
            collate_fn=collate_fn,
        )
        self.test_dl = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=db_num_workers(),
            collate_fn=collate_fn,
        )

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

        # Parse all annotations
        self.im_paths = []
        self.anno_paths = []
        self.anno_bboxes = []
        for anno_idx, anno_filename in enumerate(anno_filenames):
            anno_path = self.root / self.anno_dir / str(anno_filename)
            assert os.path.exists(
                anno_path
            ), f"Cannot find annotation file: {anno_path}"
            anno_bboxes, im_path = parse_pascal_voc_anno(anno_path)

            # TODO For now, ignore all images without a single bounding box in
            #      it, otherwise throws error during training.
            if len(anno_bboxes) == 0:
                continue

            if self.im_dir is None:
                self.im_paths.append(im_path)
            else:
                self.im_paths.append(im_paths[anno_idx])
            self.anno_paths.append(anno_path)
            self.anno_bboxes.append(anno_bboxes)
        assert len(self.im_paths) == len(self.anno_paths)

        # Get list of all labels
        labels = []
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                labels.append(anno_bbox.label_name)
        self.labels = list(set(labels))

        # Set for each bounding box label name also what its integer
        # representation is
        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                anno_bbox.label_idx = (
                    self.labels.index(anno_bbox.label_name) + 1
                )

    def split_train_test(
        self, train_pct: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set

        Args:
            train_pct: the ratio of images to use for training vs

        Return
            A training and testing dataset in that order
        """
        # TODO Is it possible to make these lines in split_train_test() a bit
        #      more intuitive?

        test_num = math.floor(len(self) * (1 - train_pct))
        if self.seed:
            torch.manual_seed(self.seed)
        indices = torch.randperm(len(self)).tolist()

        train_idx = indices[test_num:]
        test_idx = indices[: test_num + 1]

        # indicate whether the data are for training or testing
        self.is_test = np.zeros((len(self),), dtype=np.bool)
        self.is_test[test_idx] = True

        train = Subset(self, train_idx)
        test = Subset(self, test_idx)

        return train, test

    def _get_transforms(self, idx):
        """ Return the corresponding transforms for training and testing data. """
        return self.transforms[self.is_test[idx]]

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
        plot_grid(display_bboxes, self._get_random_anno, rows=rows, cols=cols)

    def _get_random_anno(
        self
    ) -> Tuple[List[AnnotationBbox], Union[str, Path]]:
        """ Get random annotation and corresponding image

        Returns a list of annotations and the image path
        """
        idx = random.randrange(len(self.anno_paths))
        return self.anno_bboxes[idx], self.im_paths[idx]

    def __getitem__(self, idx):
        """ Make iterable. """
        # get box/labels from annotations
        anno_bboxes = self.anno_bboxes[idx]
        boxes = [anno_bbox.rect() for anno_bbox in anno_bboxes]
        labels = [anno_bbox.label_idx for anno_bbox in anno_bboxes]

        # get area for evaluation with the COCO metric, to separate the
        # metric scores between small, medium and large boxes.
        area = [b.surface_area() for b in anno_bboxes]

        # setup target dic
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            # unique id
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            # suppose all instances are not crowd (torchvision specific)
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # get image
        im = Image.open(self.im_paths[idx]).convert("RGB")

        # and apply transforms if any
        if self.transforms:
            im, target = self._get_transforms(idx)(im, target)

        return im, target

    def __len__(self):
        return len(self.anno_paths)


class PennFudanDataset(DetectionDataset):
    """ PennFudan dataset.

    Adapted from
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
        self,
        anno_dir: str = "PedMasks",
        im_dir: str = "PNGImages",
        **kwargs,
    ):
        self.SIZE = 10
        super().__init__(
            Path(unzip_url(Urls.penn_fudan_ped_path, exist_ok=True)),
            anno_dir=anno_dir,
            im_dir=im_dir,
            **kwargs
        )

    def _read_annos(self) -> None:
        # list of images and their masks
        self.im_paths = get_files_in_directory(self.root / self.im_dir)
        self.im_paths = self.im_paths[:self.SIZE]
        self.anno_paths = get_files_in_directory(self.root / self.anno_dir)
        self.anno_paths = self.anno_paths[:self.SIZE]
        self.anno_bboxes = [
            AnnotationBbox.from_mask(
                mask,
                label_idx=1,
                label_name="person") for mask
            in self.anno_paths
        ]
        # there is only one class except background: person, indexed at 1
        self.labels = ["person"]

    def show_ims(self, rows: int = 1, cols: int = 3, seed: int = None) -> None:
        if seed or self.seed:
            random.seed(seed or self.seed)
        plot_grid(
            display_bbox_mask,
            self._get_random_anno,
            rows=rows,
            cols=cols)

    def _get_random_anno(
        self
    ) -> Tuple[Union[str, Path], Union[str, Path], List[AnnotationBbox]]:
        idx = random.randrange(len(self.anno_paths))
        return self.im_paths[idx], self.anno_paths[idx], self.anno_bboxes[idx]

    def __getitem__(self, idx):
        # get binary masks for the instances in the image
        binary_masks = binarise_mask(Image.open(self.anno_paths[idx]))

        # get the bounding rectangle for each instance
        rects = [b.rect() for b in self.anno_bboxes[idx]]
        areas = [b.surface_area() for b in self.anno_bboxes[idx]]

        # construct target
        target = {
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "boxes": torch.as_tensor(rects, dtype=torch.float32),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            # suppose all instances are not crowd
            "iscrowd": torch.zeros((len(areas),), dtype=torch.int64),
            # there is only one class: person, indexed at 1
            "labels": torch.ones((len(areas),), dtype=torch.int64),
            "masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
        }

        # load the image
        im = Image.open(self.im_paths[idx]).convert("RGB")

        # image pre-processing if needed
        if self.transforms is not None:
            im, target = self._get_transforms(idx)(im, target)

        return im, target


class COCOInstancesVal2017(DetectionDataset):
    """ COCO Instances Val 2017 dataset.

    Annotation URL: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    Annotation Path: annotations/instances_val2017.json
    Image URL: http://images.cocodataset.org/zips/val2017.zip
    Image Path: val2017/file_name
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.SIZE = 20
        anno_dir = Path(unzip_url(
            Urls.coco_val2017_annotation_path,
            exist_ok=True)).parent / "annotations"
        im_dir = Path(unzip_url(Urls.coco_val2017_image_path, exist_ok=True))
        self.anno_file = anno_dir / "instances_val2017.json"
        super().__init__(
            data_path(),
            anno_dir=anno_dir,
            im_dir=im_dir,
            **kwargs
        )

    def _read_annos(self) -> None:
        from pycocotools.coco import COCO
        self.coco = COCO(self.anno_file)
        img_ids = list(self.coco.imgs.keys())
        self.im_paths = img_ids[:self.SIZE]
        self.anno_paths = [self.coco.imgToAnns[i] for i in img_ids][:self.SIZE]
        valid_idxes = [i for i, x in enumerate(self.anno_paths) if len(x) != 0]
        self.im_paths = [self.im_paths[i] for i in valid_idxes]
        self.anno_paths = [self.anno_paths[i] for i in valid_idxes]
        self.anno_bboxes = [
            [
                AnnotationBbox.from_array_xywh(
                    anno["bbox"],
                    label_idx=None,
                    label_name=coco_labels()[anno["category_id"]]
                ) for anno in annos
            ] for annos in self.anno_paths
        ]
        self.labels = list(set([
            bbox.label_name for bboxes in self.anno_bboxes for bbox in bboxes
        ]))

        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                anno_bbox.label_idx = (
                    self.labels.index(anno_bbox.label_name) + 1
                )

    def show_ims(self, rows: int = 1, cols: int = 3, seed: int = None) -> None:
        if seed or self.seed:
            random.seed(seed or self.seed)
        plot_grid(
            display_bbox_mask,
            self._get_random_anno,
            rows=rows,
            cols=cols)

    def _get_im_path(self, idx: int) -> Union[str, Path]:
        filename = self.coco.imgs[self.im_paths[idx]]["file_name"]
        im_path = Path(self.im_dir) / filename
        return im_path

    def _get_mask(self, idx: int) -> np.ndarray:
        return np.array([
            self.coco.annToMask(anno) for anno in self.anno_paths[idx]
        ])

    def _get_random_anno(
        self
    ) -> Tuple[Union[str, Path], np.ndarray, List[AnnotationBbox]]:
        idx = random.randrange(len(self.anno_paths))
        im_path = self._get_im_path(idx)
        mask = self._get_mask(idx)
        bboxes = self.anno_bboxes[idx]
        return im_path, mask, bboxes

    def __getitem__(self, idx):
        # get binary masks for the instances in the image
        binary_masks = self._get_mask(idx)

        # get the bounding rectangle for each instance
        bboxes = self.anno_bboxes[idx]
        areas = [b.surface_area() for b in bboxes]
        rects = [b.rect() for b in bboxes]
        labels = [b.label_idx for b in bboxes]

        # construct target
        target = {
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "boxes": torch.as_tensor(rects, dtype=torch.float32),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            # TODO: Need to deal with iscrowd
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
        }

        # load the image
        img = Image.open(self._get_im_path(idx)).convert("RGB")

        # image pre-processing if needed
        if self.transforms is not None:
            img, target = self._get_transforms(idx)(img, target)

        return img, target


class COCOPersonKeypointsVal2017(DetectionDataset):
    """ COCO Instances Val 2017 dataset.

    Annotation URL: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    Annotation Path: annotations/person_keypoints_val2017.json
    Image URL: http://images.cocodataset.org/zips/val2017.zip
    Image Path: val2017/file_name
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.SIZE = 50
        anno_dir = Path(unzip_url(
            Urls.coco_val2017_annotation_path,
            exist_ok=True)).parent / "annotations"
        im_dir = Path(unzip_url(Urls.coco_val2017_image_path, exist_ok=True))
        self.anno_file = anno_dir / "person_keypoints_val2017.json"
        super().__init__(
            data_path(),
            anno_dir=anno_dir,
            im_dir=im_dir,
            **kwargs
        )

    def _read_annos(self) -> None:
        from pycocotools.coco import COCO
        self.coco = COCO(self.anno_file)
        img_ids = list(self.coco.imgs.keys())
        self.im_paths = img_ids[:self.SIZE]
        self.anno_paths = [self.coco.imgToAnns[i] for i in img_ids][:self.SIZE]
        valid_idxes = [i for i, x in enumerate(self.anno_paths) if len(x) != 0]
        self.im_paths = [self.im_paths[i] for i in valid_idxes]
        self.anno_paths = [self.anno_paths[i] for i in valid_idxes]
        self.anno_bboxes = [
            [
                AnnotationBbox.from_array_xywh(
                    anno["bbox"],
                    label_idx=None,
                    label_name=coco_labels()[anno["category_id"]]
                ) for anno in annos
            ] for annos in self.anno_paths
        ]
        self.labels = list(set([
            bbox.label_name for bboxes in self.anno_bboxes for bbox in bboxes
        ]))

        for anno_bboxes in self.anno_bboxes:
            for anno_bbox in anno_bboxes:
                anno_bbox.label_idx = (
                        self.labels.index(anno_bbox.label_name) + 1
                )

    def show_ims(self, rows: int = 1, cols: int = 3, seed: int = None) -> None:
        if seed or self.seed:
            random.seed(seed or self.seed)
        plot_grid(
            display_bbox_mask_keypoint,
            self._get_random_anno,
            rows=rows,
            cols=cols)

    def _get_im_path(self, idx: int) -> Union[str, Path]:
        filename = self.coco.imgs[self.im_paths[idx]]["file_name"]
        im_path = Path(self.im_dir) / filename
        return im_path

    def _get_mask(self, idx: int) -> np.ndarray:
        return np.array([
            self.coco.annToMask(anno)
            for anno in self.anno_paths[idx]
        ])

    def _get_keypoints(self, idx: int) -> np.ndarray:
        keypoints = np.array([
            anno["keypoints"] for anno in self.anno_paths[idx]
        ])
        keypoints = keypoints.reshape((len(keypoints), -1, 3)) if keypoints.size else None
        return keypoints

    def _get_random_anno(
        self
    ) -> Tuple[Union[str, Path], np.ndarray, List[AnnotationBbox], np.ndarray]:
        idx = random.randrange(len(self.anno_paths))
        print(idx)
        im_path = self._get_im_path(idx)
        mask = self._get_mask(idx)
        bboxes = self.anno_bboxes[idx]
        keypoints = self._get_keypoints(idx)
        return im_path, mask, bboxes, keypoints

    def __getitem__(self, idx):
        # get binary masks for the instances in the image
        binary_masks = self._get_mask(idx)

        # get the bounding rectangle for each instance
        bboxes = self.anno_bboxes[idx]
        areas = [b.surface_area() for b in bboxes]
        rects = [b.rect() for b in bboxes]
        labels = [b.label_idx for b in bboxes]

        # get the keypoints
        keypoints = self._get_keypoints(idx)

        # construct target
        target = {
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "boxes": torch.as_tensor(rects, dtype=torch.float32),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            # TODO: Need to deal with iscrowd
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
        }
        if keypoints:
            target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)

        # load the image
        img = Image.open(self._get_im_path(idx)).convert("RGB")

        # image pre-processing if needed
        if self.transforms is not None:
            img, target = self._get_transforms(idx)(img, target)

        return img, target
