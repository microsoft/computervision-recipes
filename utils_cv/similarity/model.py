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
    feats = featurizer.features[:]
    featurizer.features = None
    return feats


def compute_features(data, learn, embedding_layer):
    feat_dict = {}
    im_paths = [str(x) for x in list(data.items)]
    for im_path in im_paths:
        im = open_image(im_path, convert_mode='RGB')
        feat_dict[im_path] = compute_feature(im, learn, embedding_layer)
    return feat_dict

def compute_features_batched(data, learn, embedding_layer):
    error("Looks like there is a bug below")
    featurizer = SaveFeatures(embedding_layer)
    featurizer.features = None
    _ = learn.get_preds(data)
    feats = featurizer.features[:]
    im_paths = [str(x) for x in list(data.items)]
    featurizer.features = None
    return dict(zip(im_paths, feats))
