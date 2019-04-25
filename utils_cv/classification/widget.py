# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import bqplot
import bqplot.pyplot as bqpyplot
import pandas as pd
from fastai.data_block import LabelList
from ipywidgets import widgets, Layout, IntSlider
import numpy as np

from utils_cv.common.image import im_width, im_height
from utils_cv.common.data import get_files_in_directory


class AnnotationWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(
        self,
        labels: list,
        im_dir: str,
        anno_path: str,
        im_filenames: list = None,
    ):
        """Widget class to annotate images.

        Args:
            labels: List of abel names, e.g. ["bird", "car", "plane"].
            im_dir: Directory containing the images to be annotated.
            anno_path: path where to write annotations to, and (if exists) load annotations from.
            im_fnames: List of image filenames. If set to None, then will auto-detect all images in the provided image directory.
        """
        self.labels = labels
        self.im_dir = im_dir
        self.anno_path = anno_path
        self.im_filenames = im_filenames

        # Init
        self.vis_image_index = 0
        self.label_to_id = {s: i for i, s in enumerate(self.labels)}
        if not im_filenames:
            self.im_filenames = [
                os.path.basename(s)
                for s in get_files_in_directory(
                    im_dir,
                    suffixes=(
                        ".jpg",
                        ".jpeg",
                        ".tif",
                        ".tiff",
                        ".gif",
                        ".giff",
                        ".png",
                        ".bmp",
                    ),
                )
            ]
        assert (
            len(self.im_filenames) > 0
        ), f"Not a single image specified or found in directory {im_dir}."

        # Initialize empty annotations and load previous annotations if file exist
        self.annos = pd.DataFrame()
        for im_filename in self.im_filenames:
            if im_filename not in self.annos:
                self.annos[im_filename] = pd.Series(
                    {"exclude": False, "labels": []}
                )
        if os.path.exists(self.anno_path):
            print(f"Loading existing annotation from {self.anno_path}.")
            with open(self.anno_path, "r") as f:
                for line in f.readlines()[1:]:
                    vec = line.strip().split("\t")
                    im_filename = vec[0]
                    self.annos[im_filename].exclude = vec[1] == "True"
                    if len(vec) > 2:
                        self.annos[im_filename].labels = vec[2].split(",")

        # Create UI and "start" widget
        self._create_ui()

    def show(self):
        return self.ui

    def update_ui(self):
        im_filename = self.im_filenames[self.vis_image_index]
        im_path = os.path.join(self.im_dir, im_filename)

        # Update the image and info
        self.w_img.value = open(im_path, "rb").read()
        self.w_filename.value = im_filename
        self.w_path.value = self.im_dir

        # Fix the width of the image widget and adjust the height
        self.w_img.layout.height = (
            f"{int(self.IM_WIDTH * (im_height(im_path)/im_width(im_path)))}px"
        )

        # Update annotations
        self.exclude_widget.value = self.annos[im_filename].exclude
        for w in self.label_widgets:
            w.value = False
        for label in self.annos[im_filename].labels:
            label_id = self.label_to_id[label]
            self.label_widgets[label_id].value = True

    def _create_ui(self):
        """Create and initialize widgets"""
        # ------------
        # Callbacks + logic
        # ------------
        def skip_image():
            """Return true if image should be skipped, and false otherwise."""
            # See if UI-checkbox to skip images is checked
            if not self.w_skip_annotated.value:
                return False

            # Stop skipping if image index is out of bounds
            if (
                self.vis_image_index <= 0
                or self.vis_image_index >= len(self.im_filenames) - 1
            ):
                return False

            # Skip if image has annotation
            im_filename = self.im_filenames[self.vis_image_index]
            labels = self.annos[im_filename].labels
            exclude = self.annos[im_filename].exclude
            if exclude or len(labels) > 0:
                return True

            return False

        def button_pressed(obj):
            """Next / previous image button callback."""
            # Find next/previous image. Variable step is -1 or +1 depending on which button was pressed.
            step = int(obj.value)
            self.vis_image_index += step
            while skip_image():
                self.vis_image_index += step

            self.vis_image_index = min(
                max(self.vis_image_index, 0), len(self.im_filenames) - 1
            )
            self.w_image_slider.value = self.vis_image_index
            self.update_ui()

        def slider_changed(obj):
            """Image slider callback.
            Need to wrap in try statement to avoid errors when slider value is not a number.
            """
            try:
                self.vis_image_index = int(obj["new"]["value"])
                self.update_ui()
            except Exception:
                pass

        def anno_changed(obj):
            """Label checkbox callback.
            Update annotation file and write to disk
            """
            # Test if call is coming from the user having clicked on a checkbox to change its state,
            # rather than a change of state when e.g. the checkbox value was updated programatically. This is a bit
            # of hack, but necessary since widgets.Checkbox() does not support a on_click() callback or similar.
            if (
                "new" in obj
                and isinstance(obj["new"], dict)
                and len(obj["new"]) == 0
            ):
                # If single-label annotation then unset all checkboxes except the one which the user just clicked
                if not self.w_multi_class.value:
                    for w in self.label_widgets:
                        if w.description != obj["owner"].description:
                            w.value = False

                # Update annotation object
                im_filename = self.im_filenames[self.vis_image_index]
                self.annos[im_filename].labels = [
                    w.description for w in self.label_widgets if w.value
                ]
                self.annos[im_filename].exclude = self.exclude_widget.value

                # Write to disk as tab-separated file.
                with open(self.anno_path, "w") as f:
                    f.write(
                        "{}\t{}\t{}\n".format(
                            "IM_FILENAME", "EXCLUDE", "LABELS"
                        )
                    )
                    for k, v in self.annos.items():
                        if v.labels != [] or v.exclude:
                            f.write(
                                "{}\t{}\t{}\n".format(
                                    k, v.exclude, ",".join(v.labels)
                                )
                            )

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
            max=len(self.im_filenames) - 1,
            step=1,
            value=self.vis_image_index,
            continuous_update=False,
        )
        self.w_image_slider.observe(slider_changed)
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
        # Options widgets
        self.w_skip_annotated = widgets.Checkbox(
            value=False, description="Skip annotated images."
        )
        self.w_multi_class = widgets.Checkbox(
            value=False, description="Allow multi-class labeling"
        )

        # Label checkboxes widgets
        self.exclude_widget = widgets.Checkbox(
            value=False, description="EXCLUDE IMAGE"
        )
        self.exclude_widget.observe(anno_changed)
        self.label_widgets = [
            widgets.Checkbox(value=False, description=label)
            for label in self.labels
        ]
        for label_widget in self.label_widgets:
            label_widget.observe(anno_changed)

        # Combine UIs into tab widget
        w_info = widgets.VBox(
            children=[
                widgets.HTML(value="Options:"),
                self.w_skip_annotated,
                self.w_multi_class,
                widgets.HTML(value="Annotations:"),
                self.exclude_widget,
                *self.label_widgets,
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
        self.ui.set_title(0, "Annotator")

        # Fill UI with content
        self.update_ui()


class ResultsWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(self, dataset: LabelList, y_score: np.ndarray, y_label: iter):
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

    @staticmethod
    def _list_sort(list1d, reverse=False, comparison_fn=lambda x: x):
        """Sorts list1f and returns (sorted list, list of indices)"""
        indices = list(range(len(list1d)))
        tmp = sorted(zip(list1d, indices), key=comparison_fn, reverse=reverse)
        return list(map(list, list(zip(*tmp))))

    def show(self):
        return self.ui

    def update(self):
        scores = self.pred_scores[self.vis_image_index]
        im = self.dataset.x[self.vis_image_index]  # fastai Image object

        _, sort_order = self._list_sort(scores, reverse=True)
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
