from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..datasets.coco import CocoDataset
from ..datasets.coco_utils import annotation_to_mask_array


def annotation_size_distribution(
    category: str, train_annotation_sizes, val_annotation_sizes, filter_func: Callable
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    def filter_annotations(annotation_sizes):
        annotation_sizes = list(filter(filter_func, annotation_sizes))
        annotation_sizes = [y[1] for y in annotation_sizes]
        return annotation_sizes

    train_annotation_sizes = filter_annotations(train_annotation_sizes)
    val_annotation_sizes = filter_annotations(val_annotation_sizes)

    max_bin = max(max(train_annotation_sizes), max(val_annotation_sizes))
    bins = np.logspace(np.log10(10), np.log10(max_bin), 50)

    axs[0].hist(train_annotation_sizes, bins=bins)
    axs[0].set_title(f"[Train] {category} Annotation Size Distribution")
    axs[0].set_xlabel("Annotation Size")
    axs[0].set_xscale("log")
    axs[0].set_ylabel("Count")

    axs[1].hist(val_annotation_sizes, bins=bins)
    axs[1].set_title(f"[Val] {category} Annotation Size Distribution")
    axs[1].set_xlabel("Annotation Size")
    axs[1].set_xscale("log")
    axs[1].set_ylabel("Count")

    fig.show()


def get_annotation_sizes(dataset: CocoDataset):
    def f(i):
        return get_annotation_size(dataset, i)

    class_id_and_annotation_size: List[Tuple[int, int]] = Parallel(n_jobs=-1)(
        delayed(f)(i) for i in tqdm(range(len(dataset.annotations)))
    )
    return class_id_and_annotation_size


def get_annotation_size(dataset: CocoDataset, annotation_idx: int):
    """Get annotation size

    Parameters
    ----------
    dataset : CocoDataset
        Dataset with coco annotations
    annotation_idx : int
        Index for annotations

    Returns
    -------
    class_id_to_annotation_size
    """
    annotation = dataset.annotations[annotation_idx]
    image_id: int = annotation["image_id"]
    image_json = dataset.images_by_image_id[image_id]
    mask = annotation_to_mask_array(
        width=image_json["width"],
        height=image_json["height"],
        annotations=[annotation],
        classes=dataset.classes,
        annotation_format=dataset.annotation_format,
    )
    class_id_to_annotation_size = get_mask_distribution(mask)

    # Each mask should only contain the annotation and potentially background
    assert len(class_id_to_annotation_size) <= 2
    class_id_to_annotation_size = iter(class_id_to_annotation_size.items())

    # First item is background, second is our annotation
    (class_id, annotation_size) = next(class_id_to_annotation_size)
    # Double checking above claim
    if class_id == 0:
        (class_id, annotation_size) = next(class_id_to_annotation_size)

    return class_id, annotation_size


def get_mask_distribution(mask: np.ndarray):
    """Get distribution of labels (pixels) in mask

    Parameters
    ----------
    mask : np.ndarray
        Mask to get distribution for

    Returns
    -------
    class_id_to_class_counts : Dict[int, int]
        Mapping from class id to total number of pixels in mask for given class
    """
    class_id_to_class_counts = {}
    classes, class_counts = np.unique(mask, return_counts=True)
    for i in range(len(classes)):
        if classes[i] not in class_id_to_class_counts:
            class_id_to_class_counts[classes[i]] = class_counts[i]
        else:
            class_id_to_class_counts[classes[i]] += class_counts[i]
    return class_id_to_class_counts
