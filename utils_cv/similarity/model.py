# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Dict

import numpy as np

from fastai.basic_train import Learner
from fastai.vision import DatasetType, open_image
from fastai.vision.data import ImageDataBunch
from fastai.vision.image import Image
from torch.nn import Module
from torch import Tensor


class SaveFeatures:
    """Hook to save the features in the intermediate layers

    Source: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13

    Args:
        model_layer (nn.Module): Model layer

    """

    features = None

    def __init__(self, model_layer: Module):
        self.hook = model_layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module: Module, input: Tensor, output: Tensor):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


def compute_feature(
    im_or_path: Image, learn: Learner, embedding_layer: Module
) -> List[float]:
    """Compute features for a single image

    Args:
        im_or_path: Image or path to image
        learn: Trained model to use as featurizer
        embedding_layer: Number of columns on which to display the images

    Returns: DNN feature of the provided image.

    """
    if isinstance(im_or_path, str):
        im = open_image(im_or_path, convert_mode="RGB")
    else:
        im = im_or_path
    featurizer = SaveFeatures(embedding_layer)
    featurizer.features = None
    learn.predict(im)
    feats = featurizer.features[0][:]
    assert len(feats) > 1
    featurizer.features = None
    return feats


def compute_features(
    data: ImageDataBunch, learn: Learner, embedding_layer: Module
) -> List[dict]:
    """Compute features for multiple image NOT using mini-batching.

    Args:
        data: Fastai's image databunch
        learn: Trained model to use as featurizer
        embedding_layer: Number of columns on which to display the images

    Note: this function processes each image at a time and is hence slower
          compared to using mini-batches of >1.

    Returns: DNN feature of the provided image.

    """
    feat_dict = {}
    im_paths = [str(x) for x in list(data.items)]
    for im_path in im_paths:
        feat_dict[im_path] = compute_feature(im_path, learn, embedding_layer)
    return feat_dict


def compute_features_learner(
    data, dataset_type: DatasetType, learn: Learner, embedding_layer: Module
) -> List[Dict[str, np.array]]:
    """Compute features for multiple image using mini-batching.

    Use this function to featurize the training or test set of a learner

    Args:
        dataset_type: Specify train, valid or test set.
        learn: Trained model to use as featurizer
        embedding_layer: Number of columns on which to display the images

    Note: this function processes each image at a time and is hence slower
          compared to using mini-batches of >1.

    Returns: DNN feature of the provided image.

    """
    # Note: In Fastai, for DatasetType.Train, only the output of complete minibatches is computed. Ie if one has 101 images,
    # and uses a minibatch size of 16, then len(feats) is 96 and not 101. For DatasetType.Valid this is not the case,
    # and len(feats) is as expected 101. A way around this is to use DatasetType.Fix instead when referring to the training set.
    # See e.g. issue: https://forums.fast.ai/t/get-preds-returning-less-results-than-length-of-original-dataset/34148

    if dataset_type == DatasetType.Train or dataset_type == DatasetType.Fix:
        dataset_type = (
            DatasetType.Fix
        )  # Training set without shuffeling and no dropping of last batch. See note above.
        label_list = list(data.train_ds.items)
    elif dataset_type == DatasetType.Valid:
        label_list = list(data.valid_ds.items)
    elif dataset_type == DatasetType.Test:
        label_list = list(data.test_ds.items)
    else:
        raise Exception(
            "Dataset_type needs to be of type DatasetType.Train, DatasetType.Valid, DatasetType.Test or DatasetType.Fix."
        )

    # Update what data the learner object is using
    tmp_data = learn.data
    learn.data = data

    # Compute features
    featurizer = SaveFeatures(embedding_layer)
    learn.get_preds(dataset_type)
    feats = featurizer.features[:]

    # Set data back to before
    learn.data = tmp_data

    # Get corresponding image paths
    assert len(feats) == len(label_list)
    im_paths = [str(x) for x in label_list]
    return dict(zip(im_paths, feats))


# Use this function to featurize a provided dataset
# def compute_features_dl(data, device_data_loader, learn, embedding_layer):
#     featurizer = SaveFeatures(embedding_layer)
#     BUG: get_preds does not return features for all images, but only for complete mini batches
#     utils_cv.classification.model.get_preds(learn, device_data_loader)
#     feats = featurizer.features[:]
#     # Get corresponding image paths
#     im_paths = [str(x) for x in list(data.items)]
#     assert(len(feats) == len(im_paths))
#     return dict(zip(im_paths, feats))
