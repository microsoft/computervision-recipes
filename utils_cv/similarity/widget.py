# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import copy
from fastai.data_block import LabelList
from ipywidgets import widgets, Layout, IntSlider
import numpy as np


# TODO: REPLACE WITH NUMPY WHICH I BELIEVE HAS SIMILAR FUNCTION
def _list_sort(list1D, reverse=False, comparison_fct=lambda x: x):
    indices = list(range(len(list1D)))
    tmp = sorted(zip(list1D,indices), key=comparison_fct, reverse=reverse)
    list1D_sorted, sort_order = list(map(list, list(zip(*tmp))))
    return (list1D_sorted, sort_order) 


class DistanceWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(self, dataset: LabelList, distances: np.ndarray, query_im_path = None, sort = True):
        """Helper class to draw and update Image classification results widgets.

        Args:
            dataset (LabelList): Data used for prediction, containing ImageList x and CategoryList y.
            distances (np.ndarray): distance for each image to the query.
            sort (boolean): set to true to sort images by their smallest distance.
        """
        assert len(dataset) == len(distances)

        if sort:
            distances, sort_order = _list_sort(distances, reverse=False)
            dataset = copy.deepcopy(dataset) # create copy to not modify the input
            dataset.x.items = [dataset.x.items[i] for i in sort_order]
            dataset.y.items = [dataset.y.items[i] for i in sort_order]

        self.dataset = dataset
        self.distances = distances
        self.query_im_path = query_im_path
        self.vis_image_index = 0
        
        self._create_ui()

    def show(self):
        return self.ui

    def update(self):
        im = self.dataset.x[self.vis_image_index]  # fastai Image object

        self.w_image_header.value = f"Image index: {self.vis_image_index}"
        self.w_img.value = im._repr_png_()
        self.w_distance.value = "{:.2f}".format(self.distances[self.vis_image_index])
        self.w_filename.value = str(
            self.dataset.items[self.vis_image_index].name
        )
        self.w_path.value = str(
            self.dataset.items[self.vis_image_index].parent
        )

        # Fix the width of the image widget and adjust the height
        self.w_img.layout.height = (
            f"{int(self.IM_WIDTH * (im.size[0]/im.size[1]))}px"
        )

    def _create_ui(self):
        """Create and initialize widgets"""
        # ------------
        # Callbacks + logic
        # ------------
        def button_pressed(obj):
            """Next / previous image button callback."""
            step = int(obj.value)
            self.vis_image_index += step
            self.vis_image_index = min(
                max(0, self.vis_image_index), int(len(self.dataset)) - 1
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

        self.w_image_slider = IntSlider(
            min=0,
            max=len(self.dataset) - 1,
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
            ]
        )

        # ------------
        # UI - info (right side)
        # ------------
        self.w_filename = widgets.Text(
            value="", description="Filename:", layout=Layout(width="400px")
        )
        self.w_path = widgets.Text(
            value="", description="Path:", layout=Layout(width="400px")
        )
        self.w_distance = widgets.Text(
            value="", description="Distance:", layout=Layout(width="200px")
        )
        info_widgets = [widgets.HTML(value="Image:"), 
                        self.w_filename,
                        self.w_path,
                        self.w_distance]

        # Show query image if path is provided 
        if self.query_im_path:
            info_widgets.append(widgets.HTML(value="Query Image:"))
            w_query_img = widgets.Image(layout=Layout(width="200px"))
            w_query_img.value = open(self.query_im_path, "rb").read()
            info_widgets.append(w_query_img)
        
        # Combine UIs into tab widget
        w_info = widgets.VBox(
            children=info_widgets
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
