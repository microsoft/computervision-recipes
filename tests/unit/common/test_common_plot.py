# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import matplotlib.pyplot as plt
from utils_cv.common.plot import line_graph, show_ims


def test_line_graph():
    # Naive test to run the function without errors
    line_graph(
        values=[[1, 2, 3], [3, 2, 1]],
        labels=["Train", "Valid"],
        x_guides=[0, 1],
        x_name="Epoch",
        y_name="Accuracy",
    )
    plt.close()

    line_graph(
        values=[1, 2, 3],
        labels="Train",
        x_guides=[0, 1],
        x_name="Epoch",
        y_name="Accuracy",
    )
    plt.close()


def test_show_ims(tiny_ic_data_path):
    # Naive test to run the function without errors
    ims = [str(i) for i in Path(tiny_ic_data_path).glob("**/*.*")]
    show_ims(ims)
    plt.close()

    show_ims(ims, ["a"] * len(ims))
    plt.close()
