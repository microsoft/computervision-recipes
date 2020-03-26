# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List
import numpy as np
import scipy

from fastai.vision import LabelList
from .references.evaluate import evaluate_with_query_set


def vector_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    method: str = "l2",
    l2_normalize: bool = True,
) -> float:
    """Computes the distance between 2 vectors
    Inspired by https://github.com/Azure/ImageSimilarityUsingCntk=

    Args:
        vec1: First vector between which the distance will be computed
        vec2: Second vector
        method: Type of distance to be computed, e.g. "l1" or "l2"
        l2_normalize: Flag indicating whether the vectors should be normalized
        to be of unit length before the distance between them is computed

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


def evaluate(
    data: LabelList,
    features: Dict[str, np.array],
    use_rerank=False,
    rerank_k1=20,
    rerank_k2=6,
    rerank_lambda=0.3,
):
    """
    Computes rank@1 through rank@10 accuracy as well as mAP, optionally with re-ranking
    post-processor to improve accuracy (see the re-ranking implementation for more info).

    Args:
        data: Fastai's image labellist
        features: Dictionary of DNN features for each image
        use_rerank: use re-ranking
        rerank_k1, rerank_k2, rerank_lambda: re-ranking parameters
    Returns:
        rank_accs: accuracy at rank1 through rank10
        mAP: average precision

    """

    labels = np.array([data.y[i].obj for i in range(len(data.y))])
    features = np.array([features[str(s)] for s in data.items])

    # Assign each image into its own group. This serves as id during evaluation to
    # ensure a query image is not compared to itself during rank computation.
    # For the market-1501 dataset, the group ids can be used to ensure that a query
    # can not match to an image taken from the same camera.
    groups = np.array(range(len(labels)))
    assert len(labels) == len(groups) == features.shape[0]

    # Run evaluation
    rank_accs, mAP = evaluate_with_query_set(
        labels,
        groups,
        features,
        labels,
        groups,
        features,
        use_rerank,
        rerank_k1,
        rerank_k2,
        rerank_lambda,
    )
    return rank_accs, mAP
