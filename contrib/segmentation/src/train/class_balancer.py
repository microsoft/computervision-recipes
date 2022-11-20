from typing import Dict, List, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data.dataset import Dataset


def semantic_segmentation_class_balancer(dataset: Dataset) -> torch.Tensor:
    """Semantic Segmentation Class Balancer

    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset to balance classes for

    Returns
    -------
    weights : Tensor
        Tensor of size C corresponding to the number of classes
    """

    def f(i):
        class_id_to_class_counts = {}
        _, mask = dataset[i]
        # Long running function for mask size
        classes, class_counts = np.unique(mask, return_counts=True)
        size = len(mask.reshape(-1))

        for j in range(len(classes)):
            if classes[j] not in class_id_to_class_counts:
                class_id_to_class_counts[classes[j]] = class_counts[j]
            else:
                class_id_to_class_counts[classes[j]] += class_counts[j]

        return (class_id_to_class_counts, size)

    # Calculate the number of each class (by pixel count) in each mask in parallel
    class_count_info: List[Tuple[Dict, int]] = Parallel(n_jobs=-1)(
        delayed(f)(i) for i in range(len(dataset))
    )

    class_id_to_total_counts = {}
    total_size = 0

    # Synchronized summation over the class counts and size of mask
    for class_id_to_class_counts, size in class_count_info:
        for class_id, count in class_id_to_class_counts.items():
            class_id = int(class_id)
            if class_id not in class_id_to_total_counts:
                class_id_to_total_counts[class_id] = count
            else:
                class_id_to_total_counts[class_id] += count
        total_size += size

    # Normalize the class counts based on the total size
    class_id_to_total_counts = {
        class_id: count / total_size
        for class_id, count in class_id_to_total_counts.items()
    }

    # Weight scaling calculation. It should be inversely proportional to the number
    # of each class
    weights_length = int(max(class_id_to_total_counts.keys())) + 1
    weights = [0] * weights_length
    for class_id in range(weights_length):
        if class_id not in class_id_to_total_counts:
            weights[class_id] = 0
        else:
            # Weights should be scaled larger for those with lower counts
            weights[class_id] = 1 - class_id_to_total_counts[class_id]

    return torch.Tensor(weights)