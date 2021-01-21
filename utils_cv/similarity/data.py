# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random
from typing import List, Dict
from urllib.parse import urljoin

from fastai.data_block import LabelList

from utils_cv.similarity.metrics import vector_distance


class Urls:
    # base url
    base = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_similarity/"

    # traditional datasets
    fridge_objects_retrieval_path = urljoin(
        base, "fridgeObjectsImageRetrieval.zip"
    )
    fridge_objects_retrieval_tiny_path = urljoin(
        base, "fridgeObjectsImageRetrievalTiny.zip"
    )

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]


class ComparativeSet:
    """Class to represent a comparative set with a query image, 1 positive image
       and multiple negative images.

    """

    pos_dist = None
    neg_dists = None
    distance_method = "l2"

    def __init__(
        self,
        query_im_path: str,
        pos_im_path: str,
        neg_im_paths: List[str],
        pos_label: str,
        neg_labels: List[str],
    ):
        self.query_im_path = query_im_path
        self.pos_im_path = pos_im_path
        self.neg_im_paths = neg_im_paths
        self.pos_label = pos_label
        self.neg_labels = neg_labels
        assert len(neg_im_paths) > 1
        assert len(neg_im_paths) == len(neg_labels)
        assert isinstance(query_im_path, str)
        assert isinstance(pos_im_path, str)
        assert isinstance(neg_im_paths[0], str)
        assert isinstance(neg_im_paths, list)

    def __repr__(self):
        return f"ComparativeSet with {len(self.neg_im_paths)} negative images and positive label `{self.pos_label}`."

    def compute_distances(self, features: List[Dict[str, np.array]]):
        query_feature = features[self.query_im_path]
        pos_feature = features[self.pos_im_path]
        neg_features = [features[path] for path in self.neg_im_paths]
        self.pos_dist = vector_distance(
            query_feature, pos_feature, method=self.distance_method
        )
        self.neg_dists = np.array(
            [vector_distance(query_feature, f) for f in neg_features]
        )

    def set_distance_method(self, method: str):
        # for now, assert that l2 is the only distance method used.
        assert method == "l2"
        self.distance_method = method

    def pos_rank(self):
        assert self.pos_dist is not None, "Distances not computed yet."
        return sum(self.pos_dist > self.neg_dists) + 1


def comparative_set_builder(
    data: LabelList, num_sets: int, num_negatives: int = 100
) -> List[ComparativeSet]:
    """Builds sets of comparative images

    Args:
        data: Fastai's image labellist

    Returns: List of comparative_sets

    """
    random.seed(975)
    comparative_sets = []

    all_paths = [str(s) for s in list(data.x.items)]
    all_labels = [str(category.obj) for category in data.y]

    for num_set in range(num_sets):
        # Retrieve random query image
        query_index = np.random.randint(len(data))
        query_im_path = all_paths[query_index]
        query_label = all_labels[query_index]

        # List image candidates
        pos_candidates_paths = [
            all_paths[i]
            for i in range(len(all_paths))
            if (query_label == all_labels[i] and all_paths[i] != query_im_path)
        ]
        neg_candidates_indices = [
            i for i in range(len(all_paths)) if (query_label != all_labels[i])
        ]
        neg_candidates_paths = [all_paths[i] for i in neg_candidates_indices]
        neg_candidates_labels = [all_labels[i] for i in neg_candidates_indices]

        # Randomly select one positive image
        pos_index = np.random.randint(len(pos_candidates_paths))
        positive_im_path = pos_candidates_paths[pos_index]

        # Randomly select negative images
        neg_indices = np.random.randint(
            len(neg_candidates_paths), size=num_negatives
        )
        negative_im_paths = [neg_candidates_paths[i] for i in neg_indices]
        negative_labels = [neg_candidates_labels[i] for i in neg_indices]

        # Create and add comparative set to list
        comparative_set = ComparativeSet(
            query_im_path,
            positive_im_path,
            negative_im_paths,
            query_label,
            negative_labels,
        )
        comparative_sets.append(comparative_set)

    return comparative_sets
