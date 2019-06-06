# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random

from pathlib import Path

from fastai.data_block import LabelList


def comparative_set_builder(validation_bunch: LabelList) -> dict:
    """Builds sets of comparative images

    Args:
        validation_bunch: (databunch) Fast.ai's databunch containing the validation images

    Returns: comparative_sets (dict) a dictionary
    where keys are each of the images in the test_folder,
    and values are lists of 1 positive and
    (num_im_per_class x number of other classes) negative examples.
    Each key is considered as the reference image of a comparative set.

    """
    comparative_sets = dict()
    random.seed(975)

    all_classes = [
        validation_bunch.y[idx].obj for idx in range(len(validation_bunch))
    ]
    all_paths = list(validation_bunch.x.items)

    for idx in range(len(validation_bunch)):
        # ---- Extract one positive example, i.e. image from same class ----
        # Retrieve the image class name
        class_name = all_classes[idx]
        im_path = all_paths[idx]
        # List available images in the same class
        class_im_list = [
            str(all_paths[k])
            for k in range(len(all_paths))
            if (class_name == all_classes[k] and all_paths[k] != im_path)
        ]
        # Randomly select 1 positive image
        positive_index = random.sample(range(len(class_im_list)), 1)
        # Convert string into Path object to be able to apply as_posix() to it
        positive_example = Path(class_im_list[positive_index[0]]).as_posix()

        # ---- Extract all negative examples that exist in the folder ----
        negative_examples = list(
            set([str(f) for f in all_paths]).difference(set(class_im_list))
        )
        negative_indices = [
            all_paths.index(Path(neg_ex)) for neg_ex in negative_examples
        ]
        negative_examples = [
            Path(negative_examples[k]).as_posix()
            for k in range(len(negative_examples))
            if all_classes[negative_indices[k]] != class_name
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
