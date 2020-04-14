# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from fastai.vision.data import ImageDataBunch
from ipywidgets import widgets
import numpy as np
from typing import List, Dict

from utils_cv.similarity.metrics import compute_distances


def _list_sort(list1D, reverse=False, comparison_fct=lambda x: x):
    indices = list(range(len(list1D)))
    tmp = sorted(zip(list1D, indices), key=comparison_fct, reverse=reverse)
    list1D_sorted, sort_order = list(map(list, list(zip(*tmp))))
    return (list1D_sorted, sort_order)


class RetrievalWidget(object):
    def __init__(self, 
        ds: ImageDataBunch, 
        features: List[Dict[str, np.array]], 
        rows: int = 2, 
        cols: int = 5
    ):
        """Helper class to show most similar images to a query image.
           A new image can be used as query by clicking its yellow box above the image. 

        Args:
            ds: Dataset used for prediction, containing ImageList x.
            features: DNN features for each image.
            rows: number of image rows
            cols: number images per row
        """
        assert len(ds) == len(features)

        # init
        self.ds = ds
        self.features = features
        self.rows = rows
        self.cols = cols
        self.query_im_index = 1

        self._create_ui()

    def show(self):
        return self.ui

    def update(self):
        # Get the DNN feature for the query image
        query_im_path = str(self.ds.items[self.query_im_index])
        query_feature = self.features[query_im_path]

        # Compute the distances between the query and all reference images
        distances_obj = compute_distances(query_feature, self.features)

        # Get image paths and the distances sorted by smallest distance first
        im_paths = [
            str(distances_obj[i][0]) for i in range(len(distances_obj))
        ]
        distances = [
            float(distances_obj[i][1]) for i in range(len(distances_obj))
        ]
        _, sort_order = _list_sort(distances)

        # Update image grid UI
        for i in range(len(self.w_imgs)):
            w_img = self.w_imgs[i]
            w_button = self.w_buttons[i]

            if i < len(im_paths):
                img_index = sort_order[i]
                img_path = im_paths[img_index]
                distance = distances[img_index]

                w_img.layout.visibility = "visible"
                w_img.value = open(img_path, "rb").read()
                w_img.description = str(img_index)

                w_button.layout.visibility = "visible"
                w_button.value = str(img_index)
                w_button.tooltip = f"Rank {i} ({distance:.2f}), image index: {img_index}"

                if i == 0:
                    w_button.description = "Query"
                else:
                    w_button.description = f"Rank {i} ({distance:.2f})"
                
            else:
                w_img.layout.visibility = "hidden"
                w_button.layout.visibility = "hidden"

    def _create_ui(self):
        """Create and initialize widgets"""
        # ------------
        # Callbacks + logic
        # ------------
        def button_pressed(obj):
            self.query_im_index = int(obj.value)
            self.update()

        # ------------
        # UI - image grid
        # ------------
        self.w_imgs = []
        self.w_buttons = []
        w_img_buttons = []

        for i in range(self.rows * self.cols):
            # Initialize images
            w_img = widgets.Image(description="", width=180)
            self.w_imgs.append(w_img)

            # Initialize buttons
            w_button = widgets.Button(description="", value=i)
            w_button.on_click(button_pressed)
            if i == 0:
                w_button.button_style = "primary"
            else:
                w_button.button_style = "warning"
            self.w_buttons.append(w_button)

            # combine into image+button widget
            w_img_button = widgets.VBox(
                children=[w_button, w_img]
            )
            w_img_buttons.append(w_img_button)

        # Image grid
        w_grid_HBoxes = []
        for r in range(self.rows):
            hbox = widgets.HBox(
                children=[
                    w_img_buttons[r * self.cols + c] for c in range(self.cols)
                ]
            )
            hbox.layout.padding = "10px"
            w_grid_HBoxes.append(hbox)
        w_img_grid = widgets.VBox(w_grid_HBoxes)

        # Create tab
        self.ui = widgets.Tab(children=[w_img_grid])
        self.ui.set_title(0, "Image retrieval viewer")

        # Fill UI with content
        self.update()
