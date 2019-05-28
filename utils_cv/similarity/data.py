# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random

from pathlib import Path


def comparative_set_builder(test_im_list: list) -> dict:
    """Builds sets of comparative images

    Args:
        test_im_list: (list) List of paths to validation images

    Returns: comparative_sets (dict) a dictionary
    where keys are each of the images in the test_folder,
    and values are lists of 1 positive and
    (num_im_per_class x number of other classes) negative examples.
    Each key is considered as the reference image of a comparative set.

    """
    comparative_sets = dict()
    random.seed(975)
    for im_path in test_im_list:
        # ---- Extract one positive example, i.e. image from same class ----
        # Retrieve the image class name
        class_name = im_path.parts[-2]
        # List available images in the same class
        class_im_list = [
            str(f)
            for f in test_im_list
            if (class_name == f.parts[-2] and f != im_path)
        ]
        # Randomly select 1 positive image
        positive_index = random.sample(range(len(class_im_list)), 1)
        # Convert string into Path object to be able to apply as_posix() to it
        positive_example = Path(class_im_list[positive_index[0]]).as_posix()

        # ---- Extract all negative examples that exist in the folder ----
        negative_examples = list(
            set([str(f) for f in test_im_list]).difference(set(class_im_list))
        )
        negative_examples = [
            Path(neg_ex).as_posix()
            for neg_ex in negative_examples
            if class_name != Path(neg_ex).parts[-2]
        ]

        comparative_sets[im_path.as_posix()] = [
            positive_example
        ] + negative_examples

    return comparative_sets


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
