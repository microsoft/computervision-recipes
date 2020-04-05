# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scipy.spatial import distance
import numpy as np
from pytest import approx

from utils_cv.similarity.data import comparative_set_builder
from utils_cv.similarity.metrics import (
    compute_distances,
    evaluate,
    positive_image_ranks,
    recall_at_k,
    vector_distance,
)


def test_vector_distance():
    vec1 = np.array([-1, 0, 1.0])
    vec2 = np.array([1, -6.2, 2])
    assert vector_distance(vec1, vec2, "l2", l2_normalize=False) == approx(
        distance.euclidean(vec1, vec2)
    )
    vec1 = vec1 / np.linalg.norm(vec1, 2)
    vec2 = vec2 / np.linalg.norm(vec2, 2)
    assert vector_distance(vec1, vec2, "l2") == approx(
        distance.euclidean(vec1, vec2)
    )


def test_compute_distances():
    query_feature = [-1, 0.2, 2]
    feature_dict = {"a": [0, 3, 1], "b": [-2, -7.2, -3], "c": [1, 2, 3]}
    distances = compute_distances(query_feature, feature_dict)
    assert len(distances) == 3
    assert distances[1][0] == "b"
    assert vector_distance(query_feature, feature_dict["b"]) == approx(
        distances[1][1]
    )


def test_positive_image_ranks(tiny_ic_databunch):
    comparative_sets = comparative_set_builder(
        tiny_ic_databunch.valid_ds, num_sets=3, num_negatives=50
    )
    comparative_sets[0].pos_dist = 1.0
    comparative_sets[0].neg_dists = np.array([0, 5.7, 2.1])
    comparative_sets[1].pos_dist = -0.1
    comparative_sets[1].neg_dists = np.array([2.1, 0, 5.7])
    comparative_sets[2].pos_dist = -0.7
    comparative_sets[2].neg_dists = np.array([-2.1, -1.0, -0.8])
    ranks = positive_image_ranks(comparative_sets)
    assert ranks[1] == comparative_sets[1].pos_rank()
    assert ranks[0] == 2
    assert ranks[1] == 1
    assert ranks[2] == 4


def test_recall_at_k():
    rank_list = [1, 1, 2, 2, 2, 3, 4, 5, 6, 6]
    np.random.shuffle(rank_list)
    assert recall_at_k(rank_list, 0) == 0
    assert recall_at_k(rank_list, 1) == 20
    assert recall_at_k(rank_list, 2) == 50
    assert recall_at_k(rank_list, 3) == 60
    assert recall_at_k(rank_list, 6) == 100
    assert recall_at_k(rank_list, 10) == 100


def test_evaluate(tiny_ic_databunch, tiny_ic_databunch_valid_features):
    (rank_accs, mAP) = evaluate(
        tiny_ic_databunch.valid_ds,
        tiny_ic_databunch_valid_features,
        use_rerank=False,
    )
    assert 0 <= mAP <= 1.0
    assert len(rank_accs) == 6
    assert max(rank_accs) <= 1.001
    assert min(rank_accs) >= -0.001
    for i in range(len(rank_accs) - 1):
        rank_accs[i] <= rank_accs[i + 1]

    (rank_accs, ap) = evaluate(
        tiny_ic_databunch.valid_ds,
        tiny_ic_databunch_valid_features,
        use_rerank=True,
        rerank_k1=2,
        rerank_k2=3,
    )
    assert 0 <= mAP <= 1.0
    assert len(rank_accs) == 6
    assert max(rank_accs) <= 1.001
    assert min(rank_accs) >= -0.001
    for i in range(len(rank_accs) - 1):
        rank_accs[i] <= rank_accs[i + 1]
