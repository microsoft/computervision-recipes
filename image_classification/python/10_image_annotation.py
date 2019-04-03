#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>

# # Image annotation UI

# Open-source annotation software for object detection and image segmentation exist, however for image classification we were not able to find a good tool. Hence this notebook provides a simple UI to label images. Each image can be annotated as one or multiple labels, or marked as "Exclude" to indicate that the image should not be used for model trainining or evaluation.

# In[1]:


# Ensure edits to libraries are loaded and plotting is shown in the notebook.
get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


# In[2]:


import sys

sys.path.append("../")
from utils_ic.anno_utils import AnnotationWidget


# Set parameters.

# In[ ]:


IM_DIR = "../data/fridgeObjects/can"
ANNO_PATH = "cvbp_ic_annotation.csv"


# Start the UI. Set the "Allow multi-class labeling" checkbox to allow that images can be annotated with multiple labels. When in doubt what the annotation for an image should be, or for any other reason (e.g. blur or over-exposure), mark an image as "EXCLUDE". All annotations are saved to (and loaded from) a pandas dataframe with path specified in `anno_path`.
#
# <center>
# <img src="https://cvbp.blob.core.windows.net/public/images/document_images/anno_ui2.jpg" style="width: 600px;"/>
# <i>Annotation UI example</i>
# </center>

# In[ ]:


w_anno_ui = AnnotationWidget(
    labels=["can", "carton", "milk_bottle", "water_bottle"],
    im_dir=IM_DIR,
    anno_path=ANNO_PATH,
    im_filenames=None,  # Set to None to annotate all images in IM_DIR
)

display(w_anno_ui.show())


# Fast.ai supports using a dataframe as input to specify image paths and ground truth annotations. However, fast.ai expects the dataframe to follow a certain structure and does not support the exclude flag. We provide an example below which loads the annotations generated using the AnnotationWidget, returns a dataframe which fast.ai can load, and finally uses fast.ai's `from_df()` and `label_from_df()` functions to create an ImageList with ground truth labels.
#
# ```python
# import pandas as pd
# from fastai.vision import ImageList,ImageDataBunch
#
# # Load annotation, discard excluded images, and convert to format fastai expects
# annos = pd.read_pickle(ANNO_PATH)
# keys = [key for key in annos if (not annos[key].exclude and len(annos[key].labels)>0)]
# df = pd.DataFrame([(anno[0], ",".join(anno[1].labels)) for anno in annos[keys].items()],
#                   columns = ["name", "label"])
# display(df)
#
# # Example how to create an ImageList and assign labels using the dataframe. Note that the paths
# # in df are relative and hence IM_DIR needs to be provided as well to from_df().
# data = (ImageList.from_df(path=IM_DIR, df = df)
#        .split_by_rand_pct(valid_pct=0.5)
#        .label_from_df(label_delim=','))
# ```
