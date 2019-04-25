# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random

import numpy as np


def set_random_seed(s):
    """Set random seed
    """
    np.random.seed(s)
    random.seed(s)

    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
