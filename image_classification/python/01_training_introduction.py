#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>

# # Introduction to Training Image Classification Models

# In this notebook, we will give an introduction to using [fast.ai](https://www.fast.ai/) for image classification. We will use a small dataset of four differenet beverages to train and evaluate a model. We'll also cover one of the most common ways to store your data in your file system for image classification modelling.

# Check out fast.ai version.

# In[1]:


import fastai
from torch.cuda import get_device_name

print(f"Fast.ai: {fastai.__version__}")
print(get_device_name(0))


# Ensure edits to libraries are loaded and plotting is shown in the notebook.

# In[2]:


get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


# Import fastai. For now, we'll import all (`from fastai.vision import *`) so that we can easily use different utilies provided by the fastai library.

# In[3]:


import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils_ic.plot_utils import (
    ICResultsWidget,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from utils_ic.datasets import Urls, unzip_url
from fastai.vision import *
from fastai.metrics import accuracy


# Set some parameters. We'll use the `unzip_url` helper function to download and unzip our data.

# In[4]:


DATA_PATH = unzip_url(Urls.fridge_objects_path, exist_ok=True)
EPOCHS = 5
LEARNING_RATE = 1e-4
IMAGE_SIZE = 299
BATCH_SIZE = 16
ARCHITECTURE = models.resnet50


# ---

# ## 1. Prepare Image Classification Dataset

# In this notebook, we'll use a toy dataset called *Fridge Objects*, which consists of about a hundred images of milk bottle, carton, and water bottle photos taken with different backgrounds. With our helper function, the data set will be downloaded and unzip to `image_classification/data`.
#
# Let's set that directory to our `path` variable, which we'll use throughout the notebook, and checkout what's inside:

# In[5]:


path = Path(DATA_PATH)
path.ls()


# You'll notice that we have three different folders inside:
# - `/milk_bottle`
# - `/carton`
# - `/can`

# The most common data format for multiclass image classification is to have a folder titled the label with the images inside:
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
# and our data is already structured in that format!

# ## 2. Load Images

# To use fastai, we want to create `ImageDataBunch` so that the library can easily use multiple images (mini-batches) during training time. We create an ImageDataBunch by using fastai's [data_block apis](https://docs.fast.ai/data_block.html).

# In[6]:


data = (
    ImageList.from_folder(path)
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_from_folder()
    .transform(size=IMAGE_SIZE)
    .databunch(bs=BATCH_SIZE)
    .normalize(imagenet_stats)
)


# Lets take a look at our data using the databunch we created.

# In[7]:


data.show_batch(rows=3, figsize=(15, 11))


# Lets see all available classes:

# In[8]:


print(f"number of classes: {data.c}")
print(data.classes)


# We can also see how many images we have in our training and validation set.

# In[9]:


data.batch_stats


# In this notebook, we don't use test set. You can add it by using [add_test](https://docs.fast.ai/data_block.html#LabelLists.add_test). Please note that in the **fastai** framework, test datasets have no labels - this is the unknown data to be predicted. If you want to validate your model on a test dataset with labels, you probably need to use it as a validation set.

# ## 3. Train a Model

# For the model, we use a convolutional neural network (CNN). Specifically, we'll use **ResNet50** architecture. You can find more details about ResNet from [here](https://arxiv.org/abs/1512.03385).
#
# When training a model, there are many hypter parameters to select, such as the learning rate, the model architecture, layers to tune, and many more. With fastai, we can use the `create_cnn` function that allows us to specify the model architecture and performance indicator (metric). At this point, we already benefit from transfer learning since we download the parameters used to train on [ImageNet](http://www.image-net.org/).

# In[10]:


learn = cnn_learner(data, ARCHITECTURE, metrics=accuracy)


# Unfreeze our CNN since we're training all the layers.

# In[11]:


learn.unfreeze()


# We can call the `fit` function to train the dnn.

# In[12]:


learn.fit(EPOCHS, LEARNING_RATE)


# To see how our model has been trained, we can plot the accuracy and loss over the number of batches processed while training.

# In[13]:


# Plot the accuracy (metric) on validation set over the number of batches processed
learn.recorder.plot_metrics()


# In[14]:


# Plot losses on train and validation sets
learn.recorder.plot_losses()


# ## 4. Evaluate the Model

# To evaluate our model, lets take a look at the accuracy on the validation set.

# In[15]:


_, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
print(f"Accuracy on validation set: {float(metric)}")


#  You can call `get_preds` to get prediction scores (class probability distribution) for each sample as well.

# In[16]:


preds = learn.get_preds(ds_type=DatasetType.Valid)

# Get prediction scores and target labels of the validation set
pred_scores, pred_trues = [to_np(p) for p in preds]


# To see details of each sample and prediction results, we use our widget helper class `ICResultsWidget`. The widget shows each test image along with its ground truth label and model's prediction scores. We can use this tool to see how our model predicts each image and debug the model if needed.
#
# | |
# |:---:|
# |<img src="ic_widget.png" width="600"/>|
# |*Image Classification Result Widget*|

# In[17]:


results_ui = ICResultsWidget(
    dataset=learn.data.valid_ds,
    y_score=pred_scores,
    y_label=[data.classes[x] for x in np.argmax(pred_scores, axis=1)],
)
display(results_ui.show())


# Let's plot precision-recall and ROC curves for each class as well.

# In[18]:


plt.subplots(2, 2, figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_precision_recall_curve(pred_trues, pred_scores, data.classes, False)

plt.subplot(1, 2, 2)
plot_roc_curve(pred_trues, pred_scores, data.classes, False)

plt.show()


# When evaluating our results, we want to see where the model messes up, and whether or not we can do better. So we're interested in seeing images where the model predicted the image incorrectly but with high confidence (images with the highest loss).

# In[19]:


interp = ClassificationInterpretation.from_learner(learn)


# In[20]:


interp.plot_confusion_matrix()


# In[21]:


interp.plot_top_losses(9, figsize=(15, 11))


# That's pretty much it! Now you can bring your own dataset and train your model on them easily.

# In[ ]:
