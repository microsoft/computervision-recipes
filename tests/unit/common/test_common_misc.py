# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from utils_cv.common.misc import copy_files, set_random_seed


def test_set_random_seed():
    set_random_seed(1)

    
def test_copy_files(tmp):
    parent = os.path.join(tmp, "parent")
    child = os.path.join(parent, "child")
    dst = os.path.join(tmp, "dst")
    os.makedirs(parent)
    os.makedirs(child)
    os.makedirs(dst)
    
    file_in_child = Path(os.path.join(child, "file_in_child.txt"))
    file_in_child.touch()
    
    copy_files(file_in_child, dst)
    assert os.path.isfile(os.path.join(dst, "file_in_child.txt"))
    
    file_in_parent = Path(os.path.join(parent, "file_in_parent.txt"))
    file_in_parent.touch()
    
    copy_files([file_in_child, file_in_parent], dst)
    assert os.path.isfile(os.path.join(dst, "file_in_parent.txt"))
    
    # Check if the subdir is inferred
    copy_files([file_in_child, file_in_parent], dst, infer_subdir=True)
    dst_child = os.path.join(dst, "child")
    assert os.path.isdir(dst_child)
    assert os.path.isfile(os.path.join(dst_child, "file_in_child.txt"))
    assert not os.path.isfile(os.path.join(dst_child, "file_in_parent.txt"))
    
    # Check if the original files are removed
    copy_files([file_in_child, file_in_parent], dst, remove=True)
    assert not os.path.isfile(file_in_parent)
    assert not os.path.isfile(file_in_child)
