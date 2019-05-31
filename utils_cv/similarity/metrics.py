# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import scipy

from pathlib import Path


def compute_vector_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    method: str,
    boL2Normalize: bool = True,
    weights: list = [],
    bias: list = [],
    learner: list = [],
) -> float:
    """Computes the distance between 2 vectors
    Inspired by https://github.com/Azure/ImageSimilarityUsingCntk=

    Args:
        vec1: (array) First of the 2 vectors
        between which the distance will be computed
        vec2: (array) Second of these 2 vectors
        method: (str) Type of distance to be computed
        One of ["l1", "l2", "normalizedl2", "cosine", "correlation",
        "chisquared", "normalizedchisquared", "hamming",
        "mahalanobis", "weightedl1", "weightedl2", "weightedl2prob":
        boL2Normalize: (boolean) Flag indicating whether the vectors
        should be normalized before the distance between them is computed
        weights: (list of floats) Weights to assign to the vectors components
        bias: (list of floats) Biases to add to the computed distance
        learner: (model object) Model from which predictions are computed

    Returns: (float) Distance between the 2 input vectors

    """
    # Pre-processing
    if boL2Normalize:
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
    # elif method == "mahalanobis":
    #     # assumes covariance matric is provided, e..g. using:
    #     # sampleCovMat = np.cov(np.transpose(np.array(feats)))
    #     dist = scipy.spatial.distance.mahalanobis(vec1, vec2, sampleCovMat)
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


def compute_all_distances(
    query_features: np.array, feature_dict: dict, distance: str = "l2"
) -> dict:
    """Computes the distance between query_image
    and all the images present in feature_dict (query_image included)

    Args:
        query_features: (np.array) Features for the query image
        feature_dict: (dict) Dictionary of features,
        where key = image path and value = array of floats
        distance: (str) Type of distance to compute

    Returns: distances (dict) dictionary
    where key = path of each image from feature_dict,
    and value = distance between the query_image and that image

    """
    distances = {}
    for image, feature in feature_dict.items():
        distances[image] = compute_vector_distance(
            query_features, feature, distance
        )
    return distances


def sort_distances(distances: list) -> list:
    """Sorts image tuples by increasing distance

    Args:
        distances: (list) List of tuples (image path, distance to the query_image)

    Returns: distances[:top_k] (list) List of tuples of the k closest images to query_image

    """
    return sorted(distances.items(), key=lambda x: x[1])


def compute_topk_similar(
    query_features: np.array,
    feature_dict: dict,
    distance: str = "l2",
    top_k: int = 10,
) -> list:
    """Computes the distances between query_image and all other images in feature_dict
    Sorts them
    Returns the k closest

    Args:
        query_features: (np.array) Features for the query image
        feature_dict: (dict) Dictionary of features,
        where key = image path and value = array of floats
        distance: (str) Type of distance to compute, default = "l2"
        top_k: (int) Number of closest images to return, default =10
        distances: (list) List of tuples (image path, distance to the query_image)

    Returns: distances[:top_k] (list) List of tuples
    (image path, distance to the query_image)
    of the k closest images to query_image

    """
    distances = compute_all_distances(query_features, feature_dict, distance)
    distances = sort_distances(distances)
    return distances[:top_k]


def positive_image_rank_list(similarity_tuple_list: list) -> list:
    """Computes the rank of the positive example for each set of sorted images
    Returns the list of these ranks
    Args:
        similarity_tuple_list: (list) List of list of tuples
        (image path, distance to the query_image)

    Returns: (list) List of integer ranks

    """
    rank_list = []
    for similarity_tuple in similarity_tuple_list:
        # Extract the class of the reference image
        ref_class = Path(similarity_tuple[0][0]).parts[-2]
        # Find the positive example in the list of similar images
        positive_im_path = [
            x for x in [x[0] for x in similarity_tuple[1:]] if ref_class in x
        ]
        # Extract the index of the positive image
        idx = [x[0] for x in similarity_tuple].index(positive_im_path[0])
        # Append that index to the list of indices, for each of the comparative sets we have
        rank_list.append(idx)
    return rank_list


def positive_in_top_k(rank_list: list, threshold: int) -> float:
    """Computes the percentage of comparative sets
    for which the positive example's rank was better than
    or equal to the threshold

    Args:
        rank_list: (list) List of ranks of the positive example in each comparative set
        threshold: (int) Threshold below which the rank should be
        for the comparative set to be counted

    Returns: (float) Percentage of comparative sets with rank <= threshold

    """
    below_threshold = [x for x in rank_list if x <= threshold]
    percent_in_top_k = round(100.0 * len(below_threshold) / len(rank_list), 1)
    return percent_in_top_k
