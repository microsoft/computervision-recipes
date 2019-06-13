# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from fastai.vision import open_image


class SaveFeatures:
    """Hook to save the features in the intermediate layers

    Source: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13

    Args:
        model_layer (nn.Module): Model layer

    """

    features = None

    def __init__(self, model_layer):
        self.hook = model_layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


def compute_feature(im, learn, embedding_layer):
    featurizer = SaveFeatures(embedding_layer)
    featurizer.features = None
    _ = learn.predict(im)
    feats = featurizer.features[0][:]
    assert(len(feats) > 1)
    featurizer.features = None
    return feats


def compute_features(data, learn, embedding_layer):
    feat_dict = {}
    im_paths = [str(x) for x in list(data.items)]
    for im_path in im_paths:
        im = open_image(im_path, convert_mode='RGB')
        feat_dict[im_path] = compute_feature(im, learn, embedding_layer)
    return feat_dict


# def compute_features_batched1(dataset_type, learn, embedding_layer):
#     assert dataset_type==DatasetType.Valid or dataset_type==DatasetType.Train
#     featurizer = SaveFeatures(embedding_layer)
#     _ = learn.get_preds(dataset_type)
#     feats = featurizer.features[:]
#     #im_paths = [str(x) for x in list(data.items)]
#     return dict(zip(im_paths, feats))
#
#     from utils_cv.classification.model import get_preds
#
# def compute_features_batched2(data, device_data_loader, learn, embedding_layer):
#     featurizer = SaveFeatures(embedding_layer)
#     utils_cv.classification.model.get_preds(learn, device_data_loader)
#     feats = featurizer.features[:]
#
#     # Get corresponding image paths
#     im_paths = [str(x) for x in list(data.items)]
#     assert(len(feats) == len(im_paths))
#     return dict(zip(im_paths, feats))
