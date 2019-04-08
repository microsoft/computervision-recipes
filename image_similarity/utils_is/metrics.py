import scipy
import numpy as np


def compute_vector_distance(
    vec1, vec2, method, boL2Normalize=False, weights=[], bias=[], learner=[]
):
    """
    https://github.com/Azure/ImageSimilarityUsingCntk
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
        dist = chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    elif method == "mahalanobis":
        # assumes covariance matric is provided, e..g. using: sampleCovMat = np.cov(np.transpose(np.array(feats)))
        dist = scipy.spatial.distance.mahalanobis(vec1, vec2, sampleCovMat)
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


def compute_all_distances(query_image, feature_dict, distance="l2"):
    query_features = feature_dict[query_image]
    distances = {}
    for image, feature in feature_dict.items():
        distances[image] = compute_vector_distance(
            query_features, feature, distance
        )
    return distances


def sort_distances(distances):
    return sorted(distances.items(), key=lambda x: x[1])


def compute_topk_similar(query_image, feature_dict, distance="l2", top_k=10):
    distances = compute_all_distances(query_image, feature_dict, distance)
    distances = sort_distances(distances)
    return distances[:top_k]
