# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Image Classification results widget
"""
import bqplot
from bqplot import pyplot as bqpyplot
import fastai.data_block
from ipywidgets import widgets, Layout, IntSlider
import numpy as np


def _list_sort(list1d, reverse=False, comparison_fn=lambda x: x):
    """Sorts list1f and returns (sorted list, list of indices)"""
    indices = list(range(len(list1d)))
    tmp = sorted(zip(list1d, indices), key=comparison_fn, reverse=reverse)
    return list(map(list, list(zip(*tmp))))


class ResultsUI(object):
    def __init__(
        self,
        dataset: fastai.data_block.LabelList,
        pred_scores: iter,
        pred_labels: iter
    ):
        """Plot image classification results widget.

        Args:
            dataset (LabelList): Data used for prediction
            pred_scores (iterable): Prediction result scores
            pred_labels (iterable): Prediction result labels
        """
        assert (len(pred_scores) == len(pred_labels) == len(dataset))

        self.dataset = dataset
        self.pred_scores = pred_scores
        self.pred_labels = pred_labels

        # Init
        self.vis_image_index = 0
        self.labels = dataset.classes
        self.label_to_id = {s: i for i, s in enumerate(self.labels)}

        # Create UI
        self.ui = self.create_ui()

    # Update / redraw all UI elements
    def update_ui(self):
        pred_label = self.pred_labels[self.vis_image_index]
        scores = self.pred_scores[self.vis_image_index]
        im = self.dataset.x[self.vis_image_index]  # fastai Image object

        _, sort_order = _list_sort(scores, reverse=True)
        pred_labels_str = ""
        for i in sort_order:  # or may use up to 20 items e.g. [:min(20, len(sort_order))]:
            pred_labels_str += "{}({:3.1f}) \n".format(self.labels[i], scores[i])
        self.w_pred_labels.value = str(pred_labels_str)

        self.w_image_header.value = "Image index: {}".format(self.vis_image_index)
        self.w_img.value = im._repr_png_()
        self.w_gt_label.value = str(self.dataset.y[self.vis_image_index])
        self.w_pred_label.value = str(pred_label)
        self.w_pred_score.value = str(self.pred_scores[self.vis_image_index, self.label_to_id[pred_label]])
        self.w_index.value = str(self.vis_image_index)

        self.w_filename.value = str(self.dataset.items[self.vis_image_index].name)
        self.w_path.value = str(self.dataset.items[self.vis_image_index].parent)
        bqpyplot.clear()
        bqpyplot.bar(self.labels, scores, align='center', alpha=1.0, color=np.abs(scores),
                     scales={'color': bqplot.ColorScale(scheme='Blues', min=0)})

    # Create all UI elements
    def create_ui(self):

        # ------------
        # Callbacks + logic
        # ------------
        # Return if image should be shown
        def image_passes_filters(image_index):
            actual_label = str(self.dataset.y[image_index])
            bo_pred_correct = actual_label == self.pred_labels[image_index]
            if (bo_pred_correct and self.w_filter_correct.value) or (not bo_pred_correct and self.w_filter_wrong.value):
                return True
            return False

        # Next / previous image button callback
        def button_pressed(obj):
            step = int(obj.value)
            self.vis_image_index += step
            self.vis_image_index = min(max(0, self.vis_image_index), int(len(self.pred_labels)) - 1)
            while not image_passes_filters(self.vis_image_index):
                self.vis_image_index += step
                if self.vis_image_index <= 0 or self.vis_image_index >= int(len(self.pred_labels)) - 1:
                    break
            self.vis_image_index = min(max(0, self.vis_image_index), int(len(self.pred_labels)) - 1)
            self.w_image_slider.value = self.vis_image_index
            self.update_ui()

        # Image slider callback. Need to wrap in try statement to avoid errors when slider value is not a number.
        def slider_changed(obj):
            try:
                self.vis_image_index = int(obj['new']['value'])
                self.update_ui()
            except Exception as e:
                pass

        # ------------
        # UI - image + controls (left side)
        # ------------
        self.w_pred_labels = widgets.Textarea(value="", description="Predictions:") #, width='400px')
        self.w_pred_labels.layout.height = '300px'
        self.w_pred_labels.layout.width = '400px'
        
        w_next_image_button = widgets.Button(description="Image +1")
        w_next_image_button.value = "1"
        w_next_image_button.layout = Layout(width='80px')
        w_next_image_button.on_click(button_pressed)
        w_previous_image_button = widgets.Button(description="Image -1")
        w_previous_image_button.value = "-1"
        w_previous_image_button.layout = Layout(width='80px')
        w_previous_image_button.on_click(button_pressed)

        self.w_image_slider = IntSlider(min=0, max=len(self.pred_labels) - 1, step=1,
                                        value=self.vis_image_index, continuous_update=False)
        self.w_image_slider.observe(slider_changed)
        self.w_image_header = widgets.Text("", layout=Layout(width="130px"))
        self.w_img = widgets.Image()
        self.w_img.layout.width = '500px'
        w_image_with_header = widgets.VBox(children=[widgets.HBox(children=[w_previous_image_button, w_next_image_button, self.w_image_slider]),
                                                  self.w_img, self.w_pred_labels], width=520)

        # ------------
        # UI - info (right side)
        # ------------
        w_filter_header = widgets.HTML(value="Filters (use Image +1/-1 buttons for navigation):")
        self.w_filter_correct = widgets.Checkbox(value=True, description='Correct classifications')
        self.w_filter_wrong = widgets.Checkbox(value=True, description='Incorrect classifications')

        w_gt_header = widgets.HTML(value="Ground truth:")
        self.w_gt_label = widgets.Text(value="", description="Label:")

        w_pred_header = widgets.HTML(value="Prediction:")
        self.w_pred_label = widgets.Text(value="", description="Label:")
        self.w_pred_score = widgets.Text(value="", description="Score:")

        w_info_header = widgets.HTML(value="Image info:")
        self.w_index = widgets.Text(value="", description="Index:")
        self.w_filename = widgets.Text(value="", description="Name:")
        self.w_path = widgets.Text(value="", description="StoragePath:")

        w_scores_header = widgets.HTML(value="Classification scores:")
        self.w_scores = bqpyplot.figure()
        self.w_scores.layout.height = '250px'
        self.w_scores.layout.width = '370px'
        self.w_scores.fig_margin = {"top": 5, "bottom": 80, "left": 30, "right": 5}

        # Combine UIs into tab widget
        w_info_HBox = widgets.VBox(children=[w_filter_header, self.w_filter_correct, self.w_filter_wrong, w_gt_header,
                                           self.w_gt_label, w_pred_header, self.w_pred_label, self.w_pred_score,
                                           w_info_header, self.w_index, self.w_filename, self.w_path, w_scores_header,
                                           self.w_scores])
        w_info_HBox.layout.padding = '20px'
        vis_tabs_ui = widgets.Tab(children=[widgets.HBox(children=[w_image_with_header, w_info_HBox])])  # ,
        vis_tabs_ui.set_title(0, 'Results viewer')

        # Fill UI with content
        self.update_ui()

        return (vis_tabs_ui)
