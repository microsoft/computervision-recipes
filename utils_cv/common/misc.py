# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import random
import shutil
from typing import List, Union 

import numpy as np


def set_random_seed(s: int):
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

    
def copy_files(files: List[str], dst: str, infer_subdir=False, remove=False):
    """Copy list of files into destination
    """
    for f in files:
        if infer_subdir:
            d = os.path.join(dst, os.path.basename(os.path.dirname(f)))
        else:
            d = dst
            
        if not os.path.isdir(d):
            os.makedirs(d)
        shutil.copy(f, d)
        
        if remove:
            os.remove(f)
