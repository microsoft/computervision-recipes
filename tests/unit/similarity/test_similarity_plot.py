# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from utils_cv.similarity.plot import (
    plot_comparative_set,
    plot_rank_and_set_size,
)


def test_plot_rank():
    ranklist = [1, 2, 3, 2, 1, 5, 3, 5, 4]
    sets_sizes = [15, 20, 20, 20, 18, 19, 20, 15, 17, 18]
    plot_rank_and_set_size(ranklist, sets_sizes, show_set_size=True)


def test_plot_comparative_set(tiny_ic_data_path):
    compar_set = os.listdir(os.path.join(tiny_ic_data_path, "can"))
    compar_set = [os.path.join(tiny_ic_data_path, "can", im_name) for im_name in compar_set]
    compar_num = 5 if 5 < len(compar_set) else len(compar_set)
    plot_comparative_set(compar_set[0], compar_set)


# def plot_comparative_set(
#     query_im_path: str,
#     ref_im_paths: List[str],
#     num_cols: int,
#     figsize:Tuple[int,int] = None,
#     im_info_font_size: int = None,
