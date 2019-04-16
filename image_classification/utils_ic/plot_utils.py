# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper module for drawing widgets and plots
"""
import bqplot
import bqplot.pyplot as bqpyplot
import fastai.data_block
from ipywidgets import widgets, Layout, IntSlider
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def plot_pr_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: iter,
    show: bool = True,
    figsize: tuple = (12, 6),
):
    """Plot precision-recall and ROC curves .

    Currently, plots precision-recall and ROC curves.

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
        figsize (tuple): Figure size (w, h).
    """
    plt.subplots(2, 2, figsize=figsize)

    plt.subplot(1, 2, 1)
    plot_precision_recall_curve(y_true, y_score, classes, False)

    plt.subplot(1, 2, 2)
    plot_roc_curve(y_true, y_score, classes, False)

    if show:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, classes: iter, show: bool = True
):
    """Plot receiver operating characteristic (ROC) curves and ROC areas.

    If the given class labels are multi-label, it binarizes the classes and plots each ROC along with an averaged ROC.
    For the averaged ROC, micro-average is used.
    See details from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
    """
    assert (
        len(classes) == y_score.shape[1]
        if len(y_score.shape) == 2
        else len(classes) == 2
    )

    # Set random colors seed for reproducibility.
    np.random.seed(123)

    # Reference line
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    # Plot ROC curve
    if len(classes) == 2:
        # If y_score is soft-max output from a binary-class problem, we use the second node's output only.
        if len(y_score.shape) == 2:
            y_score = y_score[:, 1]
        _plot_roc_curve(y_true, y_score)
    else:
        y_true = label_binarize(y_true, classes=list(range(len(classes))))
        _plot_multi_roc_curve(y_true, y_score, classes)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower left")

    if show:
        plt.show()


def _plot_multi_roc_curve(y_true, y_score, classes):
    # Plot ROC for each class
    if len(classes) > 2:
        for i in range(len(classes)):
            _plot_roc_curve(y_true[:, i], y_score[:, i], classes[i])

    # Compute micro-average ROC curve and ROC area
    _plot_roc_curve(y_true.ravel(), y_score.ravel(), "avg")


def _plot_roc_curve(y_true, y_score, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if label == "avg":
        lw = 2
        prefix = "Averaged ROC"
    else:
        lw = 1
        prefix = "ROC" if label is None else f"ROC for {label}"

    plt.plot(
        fpr,
        tpr,
        color=_generate_color(),
        label=f"{prefix} (area = {roc_auc:0.2f})",
        lw=lw,
    )


def plot_precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray, classes: iter, show: bool = True
):
    """Plot precision-recall (PR) curves.

    If the given class labels are multi-label, it binarizes the classes and plots each PR along with an averaged PR.
    For the averaged PR, micro-average is used.
    See details from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    Args:
        y_true (np.ndarray): True class indices.
        y_score (np.ndarray): Estimated probabilities.
        classes (iterable): Class labels.
        show (bool): Show plot. Use False if want to manually show the plot later.
    """
    assert (
        len(classes) == y_score.shape[1]
        if len(y_score.shape) == 2
        else len(classes) == 2
    )

    # Set random colors seed for reproducibility.
    np.random.seed(123)

    # Plot ROC curve
    if len(classes) == 2:
        # If y_score is soft-max output from a binary-class problem, we use the second node's output only.
        if len(y_score.shape) == 2:
            y_score = y_score[:, 1]
        _plot_precision_recall_curve(
            y_true, y_score, average_precision_score(y_true, y_score)
        )
    else:
        y_true = label_binarize(y_true, classes=list(range(len(classes))))
        _plot_multi_precision_recall_curve(y_true, y_score, classes)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")

    if show:
        plt.show()


def _plot_multi_precision_recall_curve(y_true, y_score, classes):
    # Plot PR for each class
    if len(classes) > 2:
        for i in range(len(classes)):
            _plot_precision_recall_curve(
                y_true[:, i],
                y_score[:, i],
                average_precision_score(y_true[:, i], y_score[:, i]),
                classes[i],
            )

    # Plot averaged PR. A micro-average is used
    _plot_precision_recall_curve(
        y_true.ravel(),
        y_score.ravel(),
        average_precision_score(y_true, y_score, average="micro"),
        "avg",
    )


def _plot_precision_recall_curve(y_true, y_score, ap, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    if label == "avg":
        lw = 2
        prefix = "Averaged precision-recall"
    else:
        lw = 1
        prefix = (
            "Precision-recall"
            if label is None
            else f"Precision-recall for {label}"
        )

    plt.plot(
        recall,
        precision,
        color=_generate_color(),
        label=f"{prefix} (area = {ap:0.2f})",
        lw=lw,
    )


def _generate_color():
    return np.random.rand(3)


def _list_sort(list1d, reverse=False, comparison_fn=lambda x: x):
    """Sorts list1f and returns (sorted list, list of indices)"""
    indices = list(range(len(list1d)))
    tmp = sorted(zip(list1d, indices), key=comparison_fn, reverse=reverse)
    return list(map(list, list(zip(*tmp))))


class ResultsWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(
        self,
        dataset: fastai.data_block.LabelList,
        y_score: np.ndarray,
        y_label: iter,
    ):
        """Helper class to draw and update Image classification results widgets.

        Args:
            dataset (LabelList): Data used for prediction, containing ImageList x and CategoryList y.
            y_score (np.ndarray): Predicted scores.
            y_label (iterable): Predicted labels. Note, not a true label.
        """
        assert len(y_score) == len(y_label) == len(dataset)

        self.dataset = dataset
        self.pred_scores = y_score
        self.pred_labels = y_label

        # Init
        self.vis_image_index = 0
        self.labels = dataset.classes
        self.label_to_id = {s: i for i, s in enumerate(self.labels)}

        self._create_ui()

    def show(self):
        return self.ui

    def update(self):
        scores = self.pred_scores[self.vis_image_index]
        im = self.dataset.x[self.vis_image_index]  # fastai Image object

        _, sort_order = _list_sort(scores, reverse=True)
        pred_labels_str = ""
        for i in sort_order:
            pred_labels_str += f"{self.labels[i]} ({scores[i]:3.2f})\n"
        self.w_pred_labels.value = str(pred_labels_str)

        self.w_image_header.value = f"Image index: {self.vis_image_index}"

        self.w_img.value = im._repr_png_()
        # Fix the width of the image widget and adjust the height
        self.w_img.layout.height = (
            f"{int(self.IM_WIDTH * (im.size[0]/im.size[1]))}px"
        )

        self.w_gt_label.value = str(self.dataset.y[self.vis_image_index])

        self.w_filename.value = str(
            self.dataset.items[self.vis_image_index].name
        )
        self.w_path.value = str(
            self.dataset.items[self.vis_image_index].parent
        )
        bqpyplot.clear()
        bqpyplot.bar(
            self.labels,
            scores,
            align="center",
            alpha=1.0,
            color=np.abs(scores),
            scales={"color": bqplot.ColorScale(scheme="Blues", min=0)},
        )

    def _create_ui(self):
        """Create and initialize widgets"""
        # ------------
        # Callbacks + logic
        # ------------
        def image_passes_filters(image_index):
            """Return if image should be shown."""
            actual_label = str(self.dataset.y[image_index])
            bo_pred_correct = actual_label == self.pred_labels[image_index]
            if (bo_pred_correct and self.w_filter_correct.value) or (
                not bo_pred_correct and self.w_filter_wrong.value
            ):
                return True
            return False

        def button_pressed(obj):
            """Next / previous image button callback."""
            step = int(obj.value)
            self.vis_image_index += step
            self.vis_image_index = min(
                max(0, self.vis_image_index), int(len(self.pred_labels)) - 1
            )
            while not image_passes_filters(self.vis_image_index):
                self.vis_image_index += step
                if (
                    self.vis_image_index <= 0
                    or self.vis_image_index >= int(len(self.pred_labels)) - 1
                ):
                    break
            self.vis_image_index = min(
                max(0, self.vis_image_index), int(len(self.pred_labels)) - 1
            )
            self.w_image_slider.value = self.vis_image_index
            self.update()

        def slider_changed(obj):
            """Image slider callback.
            Need to wrap in try statement to avoid errors when slider value is not a number.
            """
            try:
                self.vis_image_index = int(obj["new"]["value"])
                self.update()
            except Exception:
                pass

        # ------------
        # UI - image + controls (left side)
        # ------------
        w_next_image_button = widgets.Button(description="Next")
        w_next_image_button.value = "1"
        w_next_image_button.layout = Layout(width="80px")
        w_next_image_button.on_click(button_pressed)
        w_previous_image_button = widgets.Button(description="Previous")
        w_previous_image_button.value = "-1"
        w_previous_image_button.layout = Layout(width="80px")
        w_previous_image_button.on_click(button_pressed)

        self.w_filename = widgets.Text(
            value="", description="Name:", layout=Layout(width="200px")
        )
        self.w_path = widgets.Text(
            value="", description="Path:", layout=Layout(width="200px")
        )

        self.w_image_slider = IntSlider(
            min=0,
            max=len(self.pred_labels) - 1,
            step=1,
            value=self.vis_image_index,
            continuous_update=False,
        )
        self.w_image_slider.observe(slider_changed)
        self.w_image_header = widgets.Text("", layout=Layout(width="130px"))
        self.w_img = widgets.Image()
        self.w_img.layout.width = f"{self.IM_WIDTH}px"
        w_header = widgets.HBox(
            children=[
                w_previous_image_button,
                w_next_image_button,
                self.w_image_slider,
                self.w_filename,
                self.w_path,
            ]
        )

        # ------------
        # UI - info (right side)
        # ------------
        w_filter_header = widgets.HTML(
            value="Filters (use Image +1/-1 buttons for navigation):"
        )
        self.w_filter_correct = widgets.Checkbox(
            value=True, description="Correct classifications"
        )
        self.w_filter_wrong = widgets.Checkbox(
            value=True, description="Incorrect classifications"
        )

        w_gt_header = widgets.HTML(value="Ground truth:")
        self.w_gt_label = widgets.Text(value="")
        self.w_gt_label.layout.width = "360px"

        w_pred_header = widgets.HTML(value="Predictions:")
        self.w_pred_labels = widgets.Textarea(value="")
        self.w_pred_labels.layout.height = "200px"
        self.w_pred_labels.layout.width = "360px"

        w_scores_header = widgets.HTML(value="Classification scores:")
        self.w_scores = bqpyplot.figure()
        self.w_scores.layout.height = "250px"
        self.w_scores.layout.width = "370px"
        self.w_scores.fig_margin = {
            "top": 5,
            "bottom": 80,
            "left": 30,
            "right": 5,
        }

        # Combine UIs into tab widget
        w_info = widgets.VBox(
            children=[
                w_filter_header,
                self.w_filter_correct,
                self.w_filter_wrong,
                w_gt_header,
                self.w_gt_label,
                w_pred_header,
                self.w_pred_labels,
                w_scores_header,
                self.w_scores,
            ]
        )
        w_info.layout.padding = "20px"
        self.ui = widgets.Tab(
            children=[
                widgets.VBox(
                    children=[
                        w_header,
                        widgets.HBox(children=[self.w_img, w_info]),
                    ]
                )
            ]
        )
        self.ui.set_title(0, "Results viewer")

        # Fill UI with content
        self.update()
