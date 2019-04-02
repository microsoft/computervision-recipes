# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# TODO:
# - Move helpers to utils

"""
Helper module for annotation widget
"""
import os
from ipywidgets import widgets, Layout, IntSlider
import pandas as pd
from utils_ic.common import im_width, im_height, get_files_in_directory


##################
#HELPER FUNCTIONS 
##################
class AnnotationWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(
        self,
        labels: list,
        im_dir: str,
        anno_path: str,
        im_filenames: list = None 
    ):
        """Widget class to annotate images.

        Args:
            labels: Label names.
            im_dir: Directory containing the images to be annotated.
            anno_path: path where to write annotations to, and (if exists) load annotations from. 
            im_fnames: List of image filenames. If set to None, then will auto-detect all images in the provided image directory.
        """
        if im_filenames:
            assert type(im_filenames) == list,  "Parameter im_filenames expected to be None or to be a list of strings"

        self.labels = labels
        self.im_dir = im_dir
        self.anno_path = anno_path
        self.im_filenames = im_filenames

        # Init
        self.vis_image_index = 0
        self.label_to_id = {s: i for i, s in enumerate(self.labels)}
        self.id_to_label = {i: s for i, s in enumerate(self.labels)}
        if not im_filenames:
            self.im_filenames = get_files_in_directory(im_dir, suffixes = (".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".giff", ".png", ".bmp"))
        assert len(self.im_filenames) >0, "Not a single image specified or found in directory {im_dir}."

        # Load annotations if file exist, and create empty entrees for each image 
        if not os.path.exists(self.anno_path):
            self.annos = pd.DataFrame()
        else:
            print(f"Loading existing annotation from {self.anno_path}.")
            self.annos = pd.read_pickle(self.anno_path) 
        for im_filename in self.im_filenames:
            if im_filename not in self.annos:
                self.annos[im_filename] = pd.Series({'exclude':False, 'labels':[]})
        #print(self.annos)

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
        self.w_img.layout.height = f"{int(self.IM_WIDTH * (im_height(im_path)/im_width(im_path)))}px" #     # (im.size[0]/im.size[1]))}px"
        
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
            # Stop skipping if image index is out of bounds
            if self.vis_image_index <= 0 or self.vis_image_index >= int(len(self.im_filenames)) - 1:
                return False

            # Skip if image has annotation
            im_filename = self.im_filenames[self.vis_image_index]
            labels = self.annos[im_filename].labels
            exclude = self.annos[im_filename].exclude
            if self.w_skip_annotated.value and (exclude or len(labels) > 0):
                return True

            return False

        def button_pressed(obj):
            """Next / previous image button callback."""
            # Find next/previous image
            step = int(obj.value)

            self.vis_image_index += step 
            while skip_image():
                self.vis_image_index += step
                
            self.vis_image_index = min( max(self.vis_image_index, 0), len(self.im_filenames) - 1)
            self.w_image_slider.value = self.vis_image_index
            self.update_ui()

        def slider_changed(obj):
            """Image slider callback.
            Need to wrap in try statement to avoid errors when slider value is not a number.
            """
            try:
                self.vis_image_index = int(obj['new']['value'])
                self.update_ui()
            except Exception:
                pass

        def anno_changed(obj):
            """Label checkbox callback.
            Update annotation file and write to disk
            """
            # Update annotation dictionary
            #print("Calling anno_changed obj= {}".format(obj))
            #print("*** Calling anno_changed obj[new] {}".format(obj['new']))

            #Test if calling is coming from the user having clicked on a checkbox to change its state,
            #rather than a change of state when e.g. updating the value in Python code. This is a bit of hack,
            #but necessary since widgets.Checkbox() does not support on_click() or similar.
            if 'new' in obj and obj['new']==dict() and len(obj['new']) == 0:
                im_filename = self.im_filenames[self.vis_image_index]
                label_ids = [i for i,w in enumerate(self.label_widgets) if w.value==True]

                # If single-label annotation then unset all checkboxes except the one which the user just clicked
                if not self.w_multi_class.value:
                    for w in self.label_widgets:
                        if w.description != obj['owner'].description:
                            w.value = False
                
                # Update annotations
                self.annos[im_filename].labels = [self.id_to_label[i] for i in label_ids]
                self.annos[im_filename].exclude = self.exclude_widget.value
                #print('Setting self.annos[im_filename]["exclude"] set to {}'.format(self.annos[im_filename]["exclude"]))
                #print('    and self.annos[im_filename]["labels"] set to {}'.format(self.annos[im_filename]["labels"]))
    
                # Write annotation to disk           
                keys_with_annotation = [k for k,v in self.annos.items() if (v.labels!=[] or v.exclude)]
                self.annos[keys_with_annotation].to_pickle(self.anno_path)


        # ------------
        # UI - image + controls (left side)
        # ------------
        w_next_image_button = widgets.Button(description="Next")
        w_next_image_button.value = "1"
        w_next_image_button.layout = Layout(width='80px')
        w_next_image_button.on_click(button_pressed)
        w_previous_image_button = widgets.Button(description="Previous")
        w_previous_image_button.value = "-1"
        w_previous_image_button.layout = Layout(width='80px')
        w_previous_image_button.on_click(button_pressed)

        self.w_filename = widgets.Text(value="", description="Name:", layout=Layout(width='200px'))
        self.w_path = widgets.Text(value="", description="Path:", layout=Layout(width='200px'))
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
            value=False, description='Skip annotated images.'
        )
        #self.w_skip_annotated.layout.width = '600px'
        self.w_multi_class = widgets.Checkbox(
            value=False, description='Allow multi-class labeling'
        )
        
        # Label checkboxes widgets
        self.exclude_widget = widgets.Checkbox(value=False, description="EXCLUDE IMAGE")
        self.exclude_widget.observe(anno_changed)
        self.label_widgets = [widgets.Checkbox(value=False, description=label) for label in self.labels]
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
        w_info.layout.padding = '20px'
        self.ui = widgets.Tab(
            children=[
                widgets.VBox(
                    children=[
                        w_header,
                        widgets.HBox(
                            children=[
                                self.w_img,
                                w_info,
                            ]
                        )
                    ]
                )
            ]
        )
        self.ui.set_title(0, 'Annotator')

        # Fill UI with content
        self.update_ui()
