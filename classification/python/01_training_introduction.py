#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>

# # Training an Image Classification Model
#
# In this notebook, we give an introduction to training an image classification model using [fast.ai](https://www.fast.ai/). Using a small dataset of four different beverage packages, we demonstrate training and evaluating a CNN image classification model. We also cover one of the most common ways to store data on a file system for this type of problem.
#
# ## Initialization

# In[1]:


# Ensure edits to libraries are loaded and plotting is shown in the notebook.
get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


# Import all functions we need.

# In[2]:


import sys

sys.path.append("../../")

import numpy as np
from pathlib import Path
import papermill as pm
import scrapbook as sb

# fastai and torch
import fastai
from fastai.vision import (
    models,
    ImageList,
    imagenet_stats,
    partial,
    cnn_learner,
    ClassificationInterpretation,
    to_np,
)
from fastai.metrics import accuracy

# local modules
from utils_cv.classification.model import TrainMetricsRecorder
from utils_cv.classification.plot import plot_pr_roc_curves
from utils_cv.classification.widget import ResultsWidget
from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url
from utils_cv.common.gpu import which_processor

print(f"Fast.ai version = {fastai.__version__}")
which_processor()


# This shows your machine's GPUs (if has any) and the computing device `fastai/torch` is using. We suggest using an  [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) Standard NC6 for an as needed GPU compute resource.

# Next, set some model runtime parameters. We use the `unzip_url` helper function to download and unzip the data used in this example notebook.

# In[3]:


DATA_PATH = unzip_url(Urls.fridge_objects_path, exist_ok=True)
EPOCHS = 5
LEARNING_RATE = 1e-4
IMAGE_SIZE = 299

BATCH_SIZE = 16
ARCHITECTURE = models.resnet50


# ---
#
# # Prepare Image Classification Dataset
#
# In this notebook, we use a toy dataset called *Fridge Objects*, which consists of 134 images of 4 classes of beverage container `{can, carton, milk bottle, water bottle}` photos taken on different backgrounds. The helper function downloads and unzips data set to the `image_classification/data` directory.
#
# Set that directory in the `path` variable for ease of use throughout the notebook.

# In[4]:


path = Path(DATA_PATH)
path.ls()


# You'll notice that we have four different folders inside:
# - `/water_bottle`
# - `/milk_bottle`
# - `/carton`
# - `/can`
#
# This is most common data format for multiclass image classification. Each folder title corresponds to the image label for the images contained inside:
#
# ```
# /images
# +-- can (class 1)
# |   +-- image1.jpg
# |   +-- image2.jpg
# |   +-- ...
# +-- carton (class 2)
# |   +-- image31.jpg
# |   +-- image32.jpg
# |   +-- ...
# +-- ...
# ```
#
# We have already set the data to this format structure.

# # Load Images
#
# In `fastai`, an `ImageDataBunch` can easily use multiple images (mini-batches) during training time. We create the `ImageDataBunch` by using [data_block apis](https://docs.fast.ai/data_block.html).
#
# For training and validation, we randomly split the data in an `8:2` ratio, holding 80% of the data for training and 20% for validation.
#

# In[5]:


data = (
    ImageList.from_folder(path)
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_from_folder()
    .transform(size=IMAGE_SIZE)
    .databunch(bs=BATCH_SIZE)
    .normalize(imagenet_stats)
)


# We examine some sample data using the `databunch` we created.

# In[6]:


data.show_batch(rows=3, figsize=(15, 11))


# Show all available classes:

# In[7]:


print(f"number of classes: {data.c}")
print(data.classes)


# Show the number of images in the training and validation set.

# In[8]:


data.batch_stats


# In a standard analysis, we would split the data into a train/validate/test data sets. For this example, we do not use a test set but this could be added using the [add_test](https://docs.fast.ai/data_block.html#LabelLists.add_test) method. Note that in the `fastai` framework, test sets do not include labels as this should be the unknown data to be predicted. The validation data set is a test set that includes labels that can be used to measure the model performance on new observations not used to train the model.

# # Train a Model

# For this image classifier, we use a **ResNet50** convolutional neural network (CNN) architecture. You can find more details about ResNet from [here](https://arxiv.org/abs/1512.03385).
#
# When training CNN, there are almost an infinite number of ways to construct the model architecture. We need to determine how many and what type of layers to include and how many nodes make up each layer. Other hyperparameters that control the training of those layers are also important and add to the overall complexity of neural net methods. With `fastai`, we use the `create_cnn` function to specify the model architecture and performance metric. We will use a transfer learning approach to reuse the CNN architecture and initialize the model parameters used to train on [ImageNet](http://www.image-net.org/).
#
# In this work, we use a custom callback `TrainMetricsRecorder` to track the model accuracy on the training set as we tune the model. This is for instruction only, as the standard approach in `fast.ai` [recorder class](https://docs.fast.ai/basic_train.html#Recorder) only supports tracking model accuracy on the validation set.

# In[9]:


learn = cnn_learner(
    data,
    ARCHITECTURE,
    metrics=[accuracy],
    callback_fns=[partial(TrainMetricsRecorder, show_graph=True)],
)


# Use the `unfreeze` method to allow us to retrain all the CNN layers with the <i>Fridge Objects</i> data set.

# In[10]:


learn.unfreeze()


# The `fit` function trains the CNN using the parameters specified above.

# In[11]:


learn.fit(EPOCHS, LEARNING_RATE)


# In[12]:


# You can plot loss by using the default callback Recorder.
learn.recorder.plot_losses()


# # Validate the model
#
# To validate the model, calculate the model accuracy using the validation set.

# In[13]:


_, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
print(f"Accuracy on validation set: {100*float(metric):3.2f}")


# The `ClassificationInterpretation` module is used to analyze the model classification results.

# In[14]:


interp = ClassificationInterpretation.from_learner(learn)
# Get prediction scores. We convert tensors to numpy array to plot them later.
pred_scores = to_np(interp.probs)


# To see these details use the widget helper class `ResultsWidget`. The widget shows test images along with the ground truth label and model prediction score. With this tool, it's possible to see how the model predicts each image and debug the model if needed.
#
# <img src="https://cvbp.blob.core.windows.net/public/images/ic_widget.png" width="600"/>
# <center><i>Image Classification Result Widget</i></center>

# In[15]:


w_results = ResultsWidget(
    dataset=learn.data.valid_ds,
    y_score=pred_scores,
    y_label=[data.classes[x] for x in np.argmax(pred_scores, axis=1)],
)
display(w_results.show())


# Aside from accuracy, precision and recall are other metrics that are also important in classification settings. These are linked metrics that quantify how well the model classifies an image against how it fails. Since they are linked, there is a trade-off between optimizing for precision and optimizing for recall. They can be plotted against each other to graphically show how they are linked.
#
# In multiclass settings, we plot precision-recall and [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curves for each class. In this example, the dataset is not complex and the accuracy is close to 100%. In more difficult settings, these figures will be more interesting.

# In[16]:


# True labels of the validation set. We convert to numpy array for plotting.
true_labels = to_np(interp.y_true)
plot_pr_roc_curves(true_labels, pred_scores, data.classes)


# A confusion matrix details the number of images on which the model succeeded or failed. For each class, the matrix lists correct classifications along the diagonal, and incorrect ones off-diagonal. This allows a detailed look on how the model confused the prediction of some classes.

# In[17]:


interp.plot_confusion_matrix()


# When evaluating our results, we want to see where the model makes mistakes and if we can help it improve.

# In[18]:


interp.plot_top_losses(9, figsize=(15, 11))


# In[19]:


# The following code is used by the notebook "24_run_notebook_on_azureml.ipynb" to log metrics when using papermill or scrapbook
# to run this notebook. We can comment out this cell when we are running this notebook directly.

training_losses = [x.numpy().ravel()[0] for x in learn.recorder.losses]
training_accuracy = [x[0].numpy().ravel()[0] for x in learn.recorder.metrics]

# pm.record may get deprecated and completely replaced by sb.glue:
# https://github.com/nteract/scrapbook#papermills-deprecated-record-feature
try:
    sb.glue("training_loss", training_losses)
    sb.glue("training_accuracy", training_accuracy)
    sb.glue("Accuracy on validation set:", 100 * float(metric))
except Exception:
    pm.record("training_loss", training_losses)
    pm.record("training_accuracy", training_accuracy)
    pm.record("Accuracy on validation set:", 100 * float(metric))


# # Conclusion
#
# Using the concepts introduced in this notebook, you can bring your own dataset and train an image classifier to detect objects of interest for your specific setting.
