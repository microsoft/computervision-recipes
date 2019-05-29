# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

from utils_cv.similarity.plot import (
    plot_comparative_set,
    plot_rank_and_set_size,
)
from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url


def test_plot_rank():
    ranklist = [1, 2, 3, 2, 1, 5, 3, 5, 4]
    sets_sizes = [15, 20, 20, 20, 18, 19, 20, 15, 17, 18]
    plot_rank_and_set_size(ranklist, sets_sizes, both=True)


def test_plot_comparative_set():
    fridge_object_path = unzip_url(Urls.fridge_objects_path, exist_ok=True)
    print(fridge_object_path)
    compar_set = os.listdir(os.path.join(fridge_object_path, "can"))
    compar_set = [
        os.path.join(fridge_object_path, "can", im_name)
        for im_name in compar_set
    ]
    compar_num = 5 if 5 < len(compar_set) else len(compar_set)
    plot_comparative_set(compar_set, compar_num)
    shutil.rmtree(fridge_object_path)
    os.remove(fridge_object_path + ".zip")
