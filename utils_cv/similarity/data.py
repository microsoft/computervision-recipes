# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import numpy as np
import random

from fastai.data_block import LabelList

from utils_cv.similarity.metrics import vector_distance

class ComparativeSet:
    pos_dist = None
    neg_dists = None

    def __init__(self, query_im_path, pos_im_path, neg_im_paths, pos_label, neg_labels):
        self.query_im_path = query_im_path
        self.pos_im_path = pos_im_path
        self.neg_im_paths = neg_im_paths
        self.pos_label = pos_label
        self.neg_labels = neg_labels

    def __repr__(self):
        return(f"ComparativeSet with {len(self.neg_im_paths)} negative images and positive label `{self.pos_label}`.")

    def compute_distances(self, features):
        query_feature = features[self.query_im_path]
        pos_feature = features[self.pos_im_path]
        neg_features = [features[path] for path in self.neg_im_paths]
        self.pos_dist = vector_distance(query_feature, pos_feature)
        self.neg_dists = np.array([vector_distance(query_feature, f) for f in neg_features])

    def pos_rank(self):
        assert self.pos_dist is not None, "Distances not computed yet."
        return sum(self.pos_dist > self.neg_dists)+1


def comparative_set_builder(data: LabelList) -> dict:
    """Builds sets of comparative images

    Args:
        data: (LabelList) Fast.ai's databunch containing the validation images

    Returns: comparative_sets (dict) a dictionary
    where keys are each of the images in the test_folder,
    and values are lists of 1 positive and
    (num_im_per_class x number of other classes) negative examples.
    Each key is considered as the reference image of a comparative set.

    """
    random.seed(975)
    comparative_sets = []

    all_paths = list(data.x.items)
    all_classes = [category.obj for category in data.y]

    for idx in range(len(data)):
        # ---- Extract one positive example, i.e. image from same class ----
        # Retrieve the image path and class name
        im_path = all_paths[idx]
        class_name = all_classes[idx]

        # List available images in the same class
        class_im_list = [
            str(all_paths[k])
            for k in range(len(all_paths))
            if (class_name == all_classes[k] and all_paths[k] != im_path)
        ]

        # Randomly select 1 positive image
        positive_index = random.sample(range(len(class_im_list)), 1)
        positive_example = str(Path(class_im_list[positive_index[0]]))

        # ---- Extract all negative examples that exist in the folder ----
        negative_examples = list(
            set([str(f) for f in all_paths]).difference(set(class_im_list))
        )
        negative_indices = [
            all_paths.index(Path(neg_ex)) for neg_ex in negative_examples
        ]
        negative_examples = [
            str(Path(negative_examples[k]))
            for k in range(len(negative_examples))
            if all_classes[negative_indices[k]] != class_name
        ]
        negative_labels = [
            all_classes[negative_indices[k]]
            for k in range(len(negative_examples))
        ]

        #comparative_sets[str(im_path)] = [positive_example] + negative_examples
        comparative_set = ComparativeSet(str(im_path), positive_example, negative_examples, class_name, negative_labels)
        comparative_sets.append(comparative_set)

    return comparative_sets
