# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random, pdb

from pathlib import Path

from fastai.data_block import LabelList


def comparative_set_builder(data: LabelList) -> dict:
    """Builds sets of comparative images

    Args:
        data: (LabelList) Fast.ai's databunch containing the validation images

    Returns: comparative_sets (dict) a dictionary
    where keys are each of the images in the test_folder,
    and values are lists of 1 positive and
    (num_im_per_class x number of other classes) negative examples.
    Each key is considered as the reference image of a comparative set.

    """
    random.seed(975)
    comparative_sets = dict()
    
    # all_classes = [
    #     data.y[idx].obj for idx in range(len(data))
    # ]

    all_paths = list(data.x.items)
    all_classes = [category.obj for category in data.y]
    

    for idx in range(len(data)):
        # ---- Extract one positive example, i.e. image from same class ----
        # Retrieve the image path and class name
        im_path = all_paths[idx]
        class_name = all_classes[idx]
        
        # List available images in the same class
        class_im_list = [
            str(all_paths[k])
            for k in range(len(all_paths))
            if (class_name == all_classes[k] and all_paths[k] != im_path)
        ]

        # Randomly select 1 positive image
        positive_index = random.sample(range(len(class_im_list)), 1)
        positive_example = str(Path(class_im_list[positive_index[0]])) 

        # ---- Extract all negative examples that exist in the folder ----
        negative_examples = list(
            set([str(f) for f in all_paths]).difference(set(class_im_list))
        )
        negative_indices = [
            all_paths.index(Path(neg_ex)) for neg_ex in negative_examples
        ]
        negative_examples = [
            str(Path(negative_examples[k])) 
            for k in range(len(negative_examples))
            if all_classes[negative_indices[k]] != class_name
        ]

        comparative_sets[str(im_path)] = [positive_example] + negative_examples
        # comparative_sets.append(
        #     {str(im_path): [positive_example] + negative_examples}
        # )

    return comparative_sets



