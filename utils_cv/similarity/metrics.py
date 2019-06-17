# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List

import numpy as np
import scipy


def vector_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    method: str = "l2",
    l2_normalize: bool = True,
    weights: list = [],
    bias: list = [],
    learner: list = [],
) -> float:
    """Computes the distance between 2 vectors
    Inspired by https://github.com/Azure/ImageSimilarityUsingCntk=

    Args:
        vec1: First of the 2 vectors between which the distance will be computed
        vec2: Second of these 2 vectors
        method: Type of distance to be computed, e.g. "l1" or "l2"
        l2_normalize: Flag indicating whether the vectors should be normalized
        to be of unit length before the distance between them is computed
        weights: Weights to assign to the vectors components
        bias: Biases to add to the computed distance
        learner: Model from which predictions are computed

    Returns: Distance between the 2 input vectors

    """
    # Pre-processing
    if l2_normalize:
        vec1 = vec1 / np.linalg.norm(vec1, 2)
        vec2 = vec2 / np.linalg.norm(vec2, 2)

    # Distance computation
    vecDiff = vec1 - vec2
    method = method.lower()
    if method == "l1":
        dist = sum(abs(vecDiff))
    elif method == "l2":
        dist = np.linalg.norm(vecDiff, 2)
    elif method == "normalizedl2":
        a = vec1 / np.linalg.norm(vec1, 2)
        b = vec2 / np.linalg.norm(vec2, 2)
        dist = np.linalg.norm(a - b, 2)
    elif method == "cosine":
        dist = scipy.spatial.distance.cosine(vec1, vec2)
    elif method == "correlation":
        dist = scipy.spatial.distance.correlation(vec1, vec2)
    elif method == "chisquared":
        dist = scipy.chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = scipy.chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    elif method == "weightedl1":
        feat = np.float32(abs(vecDiff))
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
        # assert(abs(dist - learnerL1.decision_function([feat])) < 0.000001)
    elif method == "weightedl2":
        feat = (vecDiff) ** 2
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
    elif method == "weightedl2prob":
        feat = (vecDiff) ** 2
        dist = learner.predict_proba([feat])[0][1]
        dist = float(dist)
    else:
        raise Exception("Distance method unknown: " + method)
    return dist


def compute_distances(
    query_feature: np.array, feature_dict: dict, method: str = "l2"
) -> List:
    """Computes the distance between query_image and all the images present in
       feature_dict (query_image included)

    Args:
        query_feature: Features for the query image
        feature_dict: Dictionary of features, where key = image path and value = array of floats
        method: distance method

    Returns: List of (image path, distance) pairs.

    """
    distances = []
    for im_path, feature in feature_dict.items():
        distance = vector_distance(query_feature, feature, method)
        distances.append((im_path, distance))
    return distances


def positive_image_ranks(comparative_sets) -> List[int]:
    """Computes the rank of the positive example for each comparative set

    Args:
        comparative_sets: List of comparative sets

    Returns: List of integer ranks

    """
    return [cs.pos_rank() for cs in comparative_sets]


def recall_at_k(ranks: List[int], k: int) -> float:
    """Computes the percentage of comparative sets where the positive image has a rank of <= k

    Args:
        ranks: List of ranks of the positive example in each comparative set
        k: Threshold below which the rank should be counted as true positive

    Returns: Percentage of comparative sets with rank <= k

    """
    below_threshold = [x for x in ranks if x <= k]
    percent_in_top_k = round(100.0 * len(below_threshold) / len(ranks), 1)
    return percent_in_top_k


# def sort_distances(distances: list) -> list:
#     """Sorts image tuples by increasing distance

#     Args:
#         distances: (list) List of tuples (image path, distance to the query_image)

#     Returns: distances[:top_k] (list) List of tuples of the k closest images to query_image

#     """
#     return sorted(distances.items(), key=lambda x: x[0])


# def compute_similars(
#     query_features: np.array, feature_dict: dict, distance: str = "l2"
# ) -> list:
#     """Computes the distances between query_image and all other images in feature_dict
#     Sorts them
#     Returns the k closest

#     Args:
#         query_features: (np.array) Features for the query image
#         feature_dict: (dict) Dictionary of features,
#         where key = image path and value = array of floats
#         distance: (str) Type of distance to compute, default = "l2"
#         top_k: (int) Number of closest images to return, default =10
#         distances: (list) List of tuples (image path, distance to the query_image)

#     Returns: distances[:top_k] (list) List of tuples
#     (image path, distance to the query_image)
#     of the k closest images to query_image

#     """
#     distances = compute_distances(query_features, feature_dict, distance)
#     distances = sort_distances(distances)
#     return distances
