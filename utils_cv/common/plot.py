# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Union

import matplotlib.pyplot as plt


def line_graph(
    values: Union[List[List[float]], List[float]],
    labels: Union[List[str], str],
    x_guides: List[int],
    x_name: str,
    y_name: str,
    legend_loc: str="lower right",
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
        plt.plot(range(values), values, label=labels, lw=1)
    else:
        assert len(values) == len(labels)
        for i, v in enumerate(values):
            plt.plot(range(len(v)), v, label=labels[i], lw=1)

    for x in x_guides:
        plt.axvline(x=x, color="gray", lw=1, linestyle="--")
        
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc=legend_loc)
