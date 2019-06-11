# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import scipy

from pathlib import Path

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
    _ = learn.predict(im)
    return featurizer.features


def compute_features(data, learn, embedding_layer):
    featurizer = SaveFeatures(embedding_layer) 
    _ = learn.get_preds(data)
    ref_features = featurizer.features
    ref_im_paths = [str(x) for x in list(data.items)]
    return dict(zip(ref_im_paths, ref_features))


