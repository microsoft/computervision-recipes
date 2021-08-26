import glob
import logging
import math
from os.path import join
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from PIL import Image

from .coco import CocoDataset

BoundingBox = Union[Tuple[int, int, int, int], List[int]]
SemanticSegmentationItem = Tuple[
    np.ndarray, List[BoundingBox], np.ndarray, List[int]
]


def construct_center_crop_on_bbox_transform(
    bbox_to_center_on: BoundingBox,
    image_dim: Tuple[int, int],
    patch_dim: Tuple[int, int] = (256, 256),
):
    """Center crop around a given bounding box

    Parameters
    ----------
    bbox_to_center_on : BoundingBox
        Bounding box to center the crop on
    image_dim : Tuple[int, int]
        Dimensions of source image `(height, width)` to take crop from
    patch_dim : Tuple[int, int]
        Patch dimension `(height, width)` to take crop on

    Returns
    -------
    transform : albumentations.Compose
        Transformation object from albumentations
    """
    height, width = image_dim
    x1, y1, x2, y2 = tuple(bbox_to_center_on)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Create coordinates for center based cropping on the annotation
    x1 = max(x1 - ((patch_dim[0] - (x2 - x1)) // 2), 0)
    x2 = min(x1 + patch_dim[0], width)
    if x2 == width:
        x1 = x2 - patch_dim[0]

    y1 = max(y1 - ((patch_dim[1] - (y2 - y1)) // 2), 0)
    y2 = min(y1 + patch_dim[1], height)
    if y2 == height:
        y1 = y2 - patch_dim[1]

    transform = A.Compose(
        [
            A.Crop(x1, y1, x2, y2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"]
        ),
    )

    return transform


def resize_until_center_crop_on_bbox(
    bbox_to_center_on: BoundingBox,
    image: np.ndarray,
    bboxes: List[BoundingBox],
    mask: np.ndarray,
    class_labels: List[int],
    resize_dim=(512, 512),
    preserve_aspect_ratio=True,
    patch_dim=(256, 256),
):
    """

    Parameters
    ----------
    bbox_to_center_on : BoundingBox
        Bounding box to center the crop on
    image : np.ndarray
        Image with shape `(height, width, channels)`
    bboxes : List[BoundingBox]
        List of bounding boxes contained in image
    mask : np.ndarray
        Segmentation mask with shape (height, width) and integer entries corresponding to the `class_labels`
    class_labels : List[int]
        List of class labels corresponding to the bounding boxes
    resize_dim : Tuple[int, int]
        `(height, width)` to resize image to before taking the crop
    preserve_aspect_ratio : bool
        True if resizing should not change the aspect ratio


    patch_dim : Tuple[int, int]
        Patch dimension `(height, width)` to take crop on
    """
    image = np.array(image)
    height, width, _ = image.shape
    x1, y1, x2, y2 = tuple(bbox_to_center_on)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if preserve_aspect_ratio:
        dim = max(resize_dim[0], resize_dim[1])
        height = height / dim
        width = width / dim
    else:
        height = resize_dim[0]
        width = resize_dim[1]

    transform = A.Compose(
        [
            A.Resize(height, width),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"]
        ),
    )
    augmented = transform(
        image=image, bboxes=bboxes, mask=mask, class_labels=class_labels
    )
    image, bboxes, mask, class_labels = (
        augmented["image"],
        augmented["bboxes"],
        augmented["mask"],
        augmented["class_labels"],
    )

    transform = construct_center_crop_on_bbox_transform(
        bbox_to_center_on,
        image_dim=(image.shape[0], image.shape[1]),
        patch_dim=patch_dim,
    )
    augmented = transform(
        image=image, bboxes=bboxes, mask=mask, class_labels=class_labels
    )
    image, bboxes, mask, class_labels = (
        augmented["image"],
        augmented["bboxes"],
        augmented["mask"],
        augmented["class_labels"],
    )

    return image, bboxes, mask, class_labels


class SemanticSegmentationResizeDataset(torch.utils.data.Dataset):
    def __init__(self, coco: CocoDataset, resize_dim: Tuple[int, int]):
        self.coco = coco
        self.resize_dim = resize_dim

    def __getitem__(self, idx) -> SemanticSegmentationItem:
        main_annotation = self.coco.annotations[idx]
        image_id: int = main_annotation["image_id"]
        (
            image,
            bboxes,
            mask,
            class_labels,
        ) = self.coco.get_semantic_segmentation_info_for_image(image_id)

        if image.shape[:2] != self.resize_dim:
            transform = A.Compose(
                [
                    A.Resize(self.resize_dim[0], self.resize_dim[1]),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["class_labels"]
                ),
            )

            augmented = transform(
                image=image,
                bboxes=bboxes,
                mask=mask,
                class_labels=class_labels,
            )

            image, bboxes, mask, class_labels = (
                augmented["image"],
                augmented["bboxes"],
                augmented["mask"],
                augmented["class_labels"],
            )

        return image, bboxes, mask, class_labels

    def __len__(self):
        return len(self.coco.annotations)


class SemanticSegmentationStochasticPatchingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patch_filepath: str,
        mask_filepath: str,
        augmentation: Callable = None,
        preprocessing: Callable = None,
    ):
        """

        Parameters
        ----------
        patch_filepath : str
            Filepath Pattern to the patch folder
        mask_filepath : str
            Filepath to the mask folder
        augmentation : Callable
            Augmentation function with signature for image, bboxes, and mask
        preprocessing : Callable
            Preprocessing function with signature for image, bboxes, and mask
        """
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.patch_paths = glob.glob(patch_filepath)
        self.mask_filepath = mask_filepath

        # self.cache: Dict[int, (np.ndarray, np.ndarray)] = {}

    def __getitem__(self, idx):
        # Cache Miss
        # if idx not in self.cache:
        patch_path = self.patch_paths[idx]
        patch_name = patch_path.split("/")[-1]
        mask_path = f"{self.mask_filepath}/{patch_name}"

        patch_image = Image.open(patch_path).convert("RGB")
        patch_image = np.array(patch_image).astype("float") / 255

        mask_image = Image.open(mask_path).convert("RGB")
        mask_image = mask_image.split()[0]  # take only red channel
        mask_image = np.array(mask_image).astype("int")

        self.cache[idx] = (patch_image, mask_image)
        # Cache Hit
        # else:
        #     patch_image, mask_image = self.cache[idx]

        # apply augmentations via albumentations
        if self.augmentation:
            sample = self.augmentation(image=patch_image, mask=mask_image)
            patch_image, mask_image = (
                sample["image"],
                sample["mask"],
            )

        # apply preprocessing
        if self.preprocessing:
            pass

        # PyTorch images should be in (channel, width, height)
        # Images are normally in (width, height, channels
        patch_image = np.moveaxis(patch_image, [0, 1, 2], [1, 2, 0])

        return (
            torch.from_numpy(patch_image).float(),
            torch.from_numpy(mask_image).long(),
        )

    def __len__(self):
        return len(self.patch_paths)


class SemanticSegmentationWithDeterministicPatchingDataset(
    torch.utils.data.Dataset
):
    def __init__(self, coco: CocoDataset, patch_dim: Tuple[int, int]):
        self.coco = coco
        self.patch_dim = patch_dim

    def __getitem__(self, annotation_idx):
        main_annotation = self.coco.annotations[annotation_idx]
        image_id: int = main_annotation["image_id"]
        (
            image,
            bboxes,
            mask,
            class_labels,
        ) = self.coco.get_semantic_segmentation_info_for_image(image_id)

        bbox_to_center_on_idx = 0
        annotations = self.coco.annotations_by_image_id[image_id]
        for idx, ann in enumerate(annotations):
            if ann["id"] == main_annotation["id"]:
                bbox_to_center_on_idx = idx
                break
        logging.info(
            f"Number of Bounding Boxes in Image for Annotation at index {annotation_idx}: {len(bboxes)}"
        )
        bbox_to_center_on: BoundingBox = bboxes[bbox_to_center_on_idx]
        transform = construct_center_crop_on_bbox_transform(
            bbox_to_center_on,
            image_dim=(image.shape[0], image.shape[1]),
            patch_dim=self.patch_dim,
        )

        augmented = transform(
            image=image, bboxes=bboxes, mask=mask, class_labels=class_labels
        )

        image, bboxes, mask, class_labels = (
            augmented["image"],
            augmented["bboxes"],
            augmented["mask"],
            augmented["class_labels"],
        )

        return image, bboxes, mask, class_labels

    def __len__(self):
        return len(self.coco.annotations)


class SemanticSegmentationDatasetFullCoverage:
    """Semantic Segmentation Dataset Strategy to cover the full image

    This dataset breaks down high resolution images into a series of cropped images.
    This allows a high resolution image that does not fit on the GPU to be loaded
    sequentially over several batches.

    This dataset is intended to be used for validation / inferencing where we would
    like to cover the entirety of the image.
    """

    def __init__(
        self,
        coco: CocoDataset,
        patch_dim: Tuple[int, int],
    ):
        self.coco = coco
        self.patch_dim = patch_dim

        # Calculate number of items that will be in the dataset
        n_images = len(self.coco.images)
        height: int = int(self.coco.images[0]["height"])
        width: int = int(self.coco.images[0]["width"])
        len_sequence = math.ceil(height / self.patch_dim[0]) * math.ceil(
            width / self.patch_dim[1]
        )
        self.length = n_images * len_sequence

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        image_idx: int = idx % len(self.coco.images)
        sequence_idx: int = int(idx / len(self.coco.images))

        image_id: int = self.coco.images[image_idx]["id"]
        (
            image,
            bboxes,
            mask,
            class_labels,
        ) = self.coco.get_semantic_segmentation_info_for_image(image_id)

        height, width, _ = image.shape
        n_windows_vertical = math.ceil(height / self.patch_dim[0])
        n_windows_horizontal = math.ceil(width / self.patch_dim[1])

        #
        x1 = self.patch_dim[1] * (sequence_idx % n_windows_horizontal)
        y1 = self.patch_dim[0] * (sequence_idx // n_windows_vertical)
        x2 = x1 + self.patch_dim[1]
        y2 = y1 + self.patch_dim[0]

        if x2 > width:
            x1 = width - self.patch_dim[1]
            x2 = width
        if y2 > height:
            y1 = height - self.patch_dim[0]
            y2 = height

        transform = A.Compose(
            [
                A.Crop(x1, y1, x2, y2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )
        augmented = transform(
            image=image, bboxes=bboxes, mask=mask, class_labels=class_labels
        )

        image, bboxes, mask, class_labels = (
            augmented["image"],
            augmented["bboxes"],
            augmented["mask"],
            augmented["class_labels"],
        )

        return image, bboxes, mask, class_labels

    def __len__(self):
        return self.length


class SemanticSegmentationDataset(torch.utils.data.Dataset):

    _available_patch_strategies = set(
        ["resize", "deterministic_center_crop", "crop_all"]
    )

    # NC24sv3 Azure VMs have 440GiB of RAM
    # This allows the SemanticSegmentationDataset to be stored in memory
    # However, when multiple workers are used in PyTorch Dataloader,
    # a separate deepcopy of the dataset is made per instance
    # Thus, disk is currently the only shared memory pool between processes
    _available_cache_strategies = set([None, "none", "disk"])

    def __init__(
        self,
        labels_filepath: str,
        classes: List[int],
        annotation_format: str,
        root_dir: str,
        cache_dir: Optional[str] = None,
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        patch_strategy: str = "deterministic_center_crop",
        patch_dim: Optional[Tuple[int, int]] = None,
        resize_dim: Optional[Tuple[int, int]] = None,
        cache_strategy: Optional[str] = None,
    ):
        if (
            patch_strategy
            not in SemanticSegmentationDataset._available_patch_strategies
        ):
            raise ValueError(
                f"Parameter `patch_strategy` must be one of {self._available_patch_strategies}"
            )

        if (
            cache_strategy
            not in SemanticSegmentationDataset._available_cache_strategies
        ):
            raise ValueError(
                f"Parameter `cache_strategy` must be one of {self._available_cache_strategies}"
            )

        if patch_strategy == "resize" and resize_dim is None:
            raise ValueError(
                'Parameter `resize_dim` must not be None if `patch_strategy` is "resize"'
            )
        elif (
            patch_strategy == "deterministic_center_crop" and patch_dim is None
        ):
            raise ValueError(
                'Parameter `patch_dim` must not be None if `patch_strategy` is "deterministic_center_crop"'
            )
        elif patch_strategy == "crop_all" and patch_dim is None:
            raise ValueError(
                'Parameter `patch_dim` must not be None if `patch_strategy is "crop_all"'
            )

        coco = CocoDataset(
            labels_filepath=labels_filepath,
            root_dir=root_dir,
            classes=classes,
            annotation_format=annotation_format,
        )

        if patch_strategy == "resize":
            self.dataset = SemanticSegmentationResizeDataset(coco, resize_dim)
        elif patch_strategy == "deterministic_center_crop":
            self.dataset = (
                SemanticSegmentationWithDeterministicPatchingDataset(
                    coco, patch_dim
                )
            )
        elif patch_strategy == "crop_all":
            self.dataset = SemanticSegmentationDatasetFullCoverage(
                coco, patch_dim
            )

        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.cache_strategy = cache_strategy

    def _get_cache_filepath_for_disk(self, idx: int):
        cache_filepath = Path(join(self.cache_dir, f"{idx}.npz"))
        return cache_filepath

    def _read_item_from_disk(self, idx: int) -> SemanticSegmentationItem:
        cache_filepath = self._get_cache_filepath_for_disk(idx)

        loaded = np.load(cache_filepath)
        image, bboxes, mask, class_labels = (
            loaded["image"],
            loaded["bboxes"],
            loaded["mask"],
            loaded["class_labels"],
        )
        return image, bboxes, mask, class_labels

    def _write_item_to_disk(
        self,
        idx: int,
        image: np.ndarray,
        bboxes: List[BoundingBox],
        mask: np.ndarray,
        class_labels: List[int],
    ):
        cache_filepath = self._get_cache_filepath_for_disk(idx)

        cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_filepath,
            image=image,
            bboxes=bboxes,
            mask=mask,
            class_labels=class_labels,
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if (
            self.cache_strategy == "disk"
            and self._get_cache_filepath_for_disk(idx).exists()
        ):
            image, bboxes, mask, class_labels = self._read_item_from_disk(idx)
        else:
            image, bboxes, mask, class_labels = self.dataset[idx]

            # Minimal memory needed
            image = image.astype("float32")
            mask = mask.astype("int8")

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = (
                    sample["image"],
                    sample["mask"],
                )

            if self.cache_strategy == "disk":
                self._write_item_to_disk(
                    idx, image, bboxes, mask, class_labels
                )

        # apply augmentations via albumentations
        if self.augmentation:
            # Currently albumentations CropNonEmptyMaskIfExists does not support bboxes so we do not use below
            # augment bboxes and class_labels
            # GitHub Issue: https://github.com/albumentations-team/albumentations/issues/461
            sample = self.augmentation(image=image, mask=mask)
            image, mask = (
                sample["image"],
                sample["mask"],
            )

        # PyTorch images should be in (channel, height, width)
        # Images are normally in (height, width, channels)
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])

        return image, mask

    def __len__(self):
        return len(self.dataset)


class ToySemanticSegmentationDataset(torch.utils.data.Dataset):
    """Toy semantic segmentation dataset for integration testing purposes"""

    def __init__(self, *args, **kwargs):
        self._dataset = SemanticSegmentationDataset(*args, **kwargs)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return min(len(self._dataset), 8)
