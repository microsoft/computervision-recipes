# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def line_graph(
    values: Union[List[List[float]], List[float]],
    labels: Union[List[str], str],
    x_guides: List[int],
    x_name: str,
    y_name: str,
    legend_loc: str = "lower right",
):
    """Plot line graph(s).

    Args:
        values: List of graphs or a graph to plot
        labels: List of labels or a label for graph.
            If labels is a string, this function assumes the values is a single graph.
        x_guides: List of guidelines (a vertical dotted line)
        x_name: x axis label
        y_name: y axis label
        legend_loc: legend location
    """
    if isinstance(labels, str):
        plt.plot(range(len(values)), values, label=labels, lw=1)
    else:
        assert len(values) == len(labels)
        for i, v in enumerate(values):
            plt.plot(range(len(v)), v, label=labels[i], lw=1)

    for x in x_guides:
        plt.axvline(x=x, color="gray", lw=1, linestyle="--")

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc=legend_loc)


def show_ims(
    im_paths: Union[str, List[str]],
    labels: Union[str, List[str]] = None,
    size: int = 3,
    rows: int = 1,
):
    """Show image files
    Args:
        im_paths (str or List[str]): Image filepaths
        labels (str or List[str]): Image labels. If None, show image file name.
        size (int): MatplotLib plot size.
        rows (int): rows of the images
    """
    if isinstance(im_paths, (str, Path)):
        if labels is not None and isinstance(labels, str):
            labels = [labels]
        ims = [mpimg.imread(im_paths)]
        im_paths = [im_paths]
    else:
        ims = [mpimg.imread(im_path) for im_path in im_paths]

    cols = math.ceil(len(ims) / rows)
    _, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.set_axis_off()

    for i, (im_path, im) in enumerate(zip(im_paths, ims)):
        if labels is None:
            axes[i].set_title(Path(im_path).stem)
        else:
            axes[i].set_title(labels[i])
        axes[i].imshow(im)
