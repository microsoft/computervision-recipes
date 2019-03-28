#!/usr/bin/env python
# coding: utf-8

# # Building Models for Accuracy VS Speed
#
# When building a deep learning model for computer vision, there are many parameters that we may want to tune in order get the kind of model we need. Sometimes, we want models that perform at the highest possible accuracy it can achieve. Other times, we want models that can be packed into small machines and optimized for mobility. By configuring these parameters, we can compose models to precisely fit our needs.
#
# ## Table of Contents:
#   * [Trade-off accuracy and speed](#introduction)
#   * [Methodology](#methodology)
#     * [Datasets](#methodology-datasets)
#     * [Model Characteristics](#methodology-model-characteristics)
#     * [Default Parameters](#methodology-default-parameters)
#   * [DNN Architecture](#dnn)
#   * [Learning Rate & Epochs](#lr)
#   * [Image Resolution](#imsize)
#   * [TLDR](#tldr)

# ---

# ## What to optimize for? <a name="introduction"></a>
#
# As practitioners of computer vision, we want to be able to control what to optimize when building our models. Unless you are building a model for a Kaggle competition, it is unlikely that you can build your model with only its accuracy in mind.
#
# In the real world, models must be able to run under varying scenarios with different constraints. Different scenarios requires us, as computer vision practitioners, to prioritize different characteristics over others.
#
# For example, in an IoT setting, where the inferencing device has limited computational capabilities, we need to design our models to have a small memory footprint. In contrast, medical situations often require the highest possible accuracy because the cost of mis-classification could impact the well-being of a patient. In this scenario, the accuracy of the model can not be compromised.
#
# This notebook will explore different characteristics when modelling and help you come up with the optimal model for your specific scenario.
#
# ## Methodology <a name="methodology"></a>
#
# To explore the charactistics of a model, we - the computer vision repo team - have conducted various experiments to explore the impact of different hyperparameters on a model's _accuracy_, _training duration_, _inference speed_, and _memory footprint_. In this notebook, we hope to outline some of the key findings so that you can make better decisions when settings parameters.
#
# In this notebook, we use the results of our experiments to give us concrete evidence when it comes to understanding which parameters work and which dont.
#
# > To recreate these experiments, you can use the `util_ic.parameter_sweeper` module that lets us easily test different parameters when building models. You can learn more about how to use the module in the [Exploring Hyperparameters](.11_exploring_hyperparamters.ipynb) notebook.
#
#
# ### Datasets <a name="methodology-datasets"></a>
#
# For our experiments, we relied on a set of six different classification datasets. (These datasets can be downloaded directly from this repo using the `util_ic.datasets` module.)
#
# | Dataset Name | Number of Images | Number of Classes |
# | --- | --- | --- |
# | food101Subset | 5000 | 5 |
# | flickrLogos32Subset | 2740 | 33 |
# | fashionTexture | 1716 | 11 |
# | recycle_v3 |  564 | 11 |
# | lettuce | 380 | 2 |
# | fridgeObjects | 134 | 4 |
#
# When selecting these datasets, we wanted to have a variety of image types with different amounts of data and number of classes.
#
#
# ### Model Characteristics <a name="methodology-model-characteristics"></a>
#
# In our experiment, we look at these characteristics to evaluate the impact of various paramters.
#
# - __Accuracy__
#
#     Accuracy is our evaluation metric for the model. It represents the average accuracy over 5 runs for our six different datasets.
#
#
# - __Training Duration__
#
#     The training duration is how long it takes to train the model. It represents the average duration over 5 runs for our six different datasets.
#
#
# - __Inference Speed__
#
#     The inference speed is the time it takes the model to run 1000 predictions.
#
#
# - __Memory Footprint__
#
#     The memory footprint is size of the model parameters saved as the pickled file. This can be achieved by running `learn.export(...)` and examining the size of the exported file.
#
# ### Default Parameters <a name="methodology-default-parameters"></a>
#
# For our experiments, we used following parameters:
#
# | Parameter | Value |
# | --- | --- |
# | Batch Size | 16 |
# | Dropout | 0.5 |
# | Weight Decay | 0.01 |
# | Momentum | 0.9 |
# | Epochs | 15 |
#
# It turns out that these parameters did not (in any significant way) affect the model's performance, training/inference speed, or memory footprint. So, for most datasets, we can use these as our default values.
#
# When running these experiments, we also set the number of epochs to 15, unless otherwise specified. For the datasets that we're using, training for 15 epochs might be more than is necessary. But one of the observations we've had is that it is extremely hard to overfit our model, so training for more epochs tends not to hurt the model's performance. However, the high number of epochs may mean that our average training durations are longer than what they could be.

# In the section below, we'll look at how different parameters affect our model.

# ## DNN Architectures <a name="dnn"></a>
#
# One of the most important decisions to make when building a model is choosing what DNN architectures to use. Some DNNs have hundreds of layers and end up having quite a large memory footprint with millions of parameters to tune, while others are compact and small enough to fit onto memory limited edge devices.
#
# When looking at an architecture, we may want to consider the characteristics mentioned above: the model's accuracy (or the model's performance metric more broadly speaking), the model's memory footprint, how long it takes to train the model, and the inference speed of the model.
#
# Lets take a __squeezenet1_1__ model, a __resnet18__ model and __resnet50__ model and compare the differences based on our experiment. For this experiment, we kept the image size at 499 pixels.
#
# ![architecture_comparisons](figs/architecture_comparisons.png)
#
# As you can see from the graphs above, there is a clear trade-off when deciding between the models.
#
# In terms of accuracy, __resnet50__ out-performs the rest, but it also suffers from having the highest memory footprint, and the longest training and inference times. On the other end of the spectrum, __squeezenet1_1__ performs the worst in terms fo accuracy, but has by far the smallest memory footprint.
#
# ---
#
# <details><summary>See the code to generate the graphs</summary>
# <p>
#
# #### Code snippet to generate graphs in this cell
#
# ```python
# import pandas as pd
# from utils_ic.parameter_sweeper import add_value_labels
# %matplotlib inline
#
# df = pd.DataFrame({
#     "accuracy": [.9472, .9190, .8251],
#     "training_duration": [385.3, 280.5, 272.5],
#     "inference_duration": [34.2, 27.8, 27.6],
#     "memory": [99, 45, 4.9],
#     "model": ['resnet50', 'resnet18', 'squeezenet1_1'],
# }).set_index("model")
#
# ax1, ax2, ax3, ax4 = df.plot.bar(
#     rot=90, subplots=True, legend=False, figsize=(8,10)
# )
#
# for ax in [ax1, ax2, ax3, ax4]:
#     for i in [0, 1, 2]:
#         if i==0: ax.get_children()[i].set_color('r')
#         if i==1: ax.get_children()[i].set_color('g')
#         if i==2: ax.get_children()[i].set_color('b')
#
# ax1.set_title("Accuracy (%)")
# ax2.set_title("Training Duration (seconds)")
# ax3.set_title("Inference Time (seconds)")
# ax4.set_title("Memory Footprint (mb)")
#
# ax1.set_ylabel("%")
# ax2.set_ylabel("seconds")
# ax3.set_ylabel("seconds")
# ax4.set_ylabel("mb")
#
# ax1.set_ylim(top=df["accuracy"].max() * 1.3)
# ax2.set_ylim(top=df["training_duration"].max() * 1.3)
# ax3.set_ylim(top=df["inference_duration"].max() * 1.3)
# ax4.set_ylim(top=df["memory"].max() * 1.3)
#
# add_value_labels(ax1, percentage=True)
# add_value_labels(ax2)
# add_value_labels(ax3)
# add_value_labels(ax4)
# ```
#
# </p>
# </details>
#

# ## Learning Rate and Epochs <a name="lr"></a>
#
# Learning rate tends to be one of the most important parameters to set when training your model.
#
# If your learning rate is set too low, training will progress very slowly since we're only making tiny updates to the weights in your network. However, if your learning rate is too high, it can cause undesirable divergent behavior in your loss function.
#
# One way to mitigate against a low learning rate is to make sure that you're training for many epochs.
#
# To efficiently build a model, we need to make sure that our learning rate is in the correct range. To find a good default learning rate, we've tested various learning rates on a range of datasets over two different epochs settings.
#
# ![lr_comparisons](figs/lr_comparisons.png)
#
# In both figures, we can see that a learning rate of 1e-3 and 1e-4 tends to work the best across the different datasets and the two settings for epochs.
#
# In the figure on the left which shows the results of the different learning rates on different datasets at 15 epochs, we can see that a learning rate of 1e-4 does the best overall. But this may not be the case for every dataset. If you look carefully, there is a pretty significant variance between the datasets and it may be possible that a learning rate of 1-e3 works better than a learning rate of 1e-4 for some datasets.
#
# In the figure on the right, both 1e-4 and 1e-3 seem to work well. At 15 epochs, the results of 1e-4 are only slightly better than that of 1e-3. However, at 3 epochs, a learning rate of 1e-3 out performs the learning rate at 1e-4. This makes sense since we're limiting the training to only 3 epochs, the model that can update its weights more quickly will perform better.
#
# As a result, we may learn towards using higher learning rates (such as 1e-3) if we want to minimize the training time, and lower learning rates (such as 1e-4) if training time is not constrained.
#
# ---
#
# <details><summary>See the code to generate the graphs</summary>
# <p>
#
# #### Code snippet to generate graphs in this cell
#
# ```python
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# df_dataset_comp = pd.DataFrame({
#     "fashionTexture": [0.8749, 0.8481, 0.2491, 0.670318, 0.1643],
#     "flickrLogos32Subset": [0.9069, 0.9064, 0.2179, 0.7175, 0.1073],
#     "food101Subset": [0.9294, 0.9127, 0.6891, 0.9090, 0.555827],
#     "fridgeObjects": [0.9591, 0.9727, 0.272727, 0.6136, 0.181818],
#     "lettuce": [0.8992, 0.9104, 0.632, 0.8192, 0.5120],
#     "recycle_v3": [0.9527, 0.9581, 0.766, 0.8591, 0.2876],
#     "learning_rate": [0.000100, 0.001000, 0.010000, 0.000010, 0.000001]
# }).set_index("learning_rate")
#
# df_epoch_comp = pd.DataFrame({
#     "3_epochs": [0.823808, 0.846394, 0.393808, 0.455115, 0.229120],
#     "15_epochs": [0.920367, 0.918067, 0.471138, 0.764786, 0.301474],
#     "learning_rate": [0.000100, 0.001000, 0.010000, 0.000010, 0.000001]
# }).set_index("learning_rate")
#
# plt.figure(1)
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
#
# vals = ax2.get_yticks()
#
# df_dataset_comp.sort_index().plot(kind='bar', rot=0, figsize=(15, 6), ax=ax1)
# vals = ax1.get_yticks()
# ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
# ax1.set_ylim(0,1)
# ax1.set_ylabel("Accuracy (%)")
# ax1.set_title("Accuracy of Learning Rates by Datasets @ 15 Epochs")
# ax1.legend(loc=2)
#
# df_epoch_comp.sort_index().plot(kind='bar', rot=0, figsize=(15, 6), ax=ax2)
# ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
# ax2.set_ylim(0,1)
# ax2.set_title("Accuracy of Learning Rates by Epochs")
# ax2.legend(loc=2)
# ```
#
# </p>
# </details>

# ## Image Resolution <a name="imsize"></a>
#
# A model's input image resolution tends to affect its accuracy. Usually, convolutional neural networks able to take advantage of higher resolution images.
#
# But how does it impact some of the other aspects of the model?
#
# It turns out that the image size doesn't affect the model's memory footprint. Because the image size doesn't change the number of parameters, it makes sense that it should not affect the model size.
#
# However, the image size has a direct impact on training and inference speeds. An increase in image size means an increase in the number of paramters we have to calculate.
#
# ![imsize_comparisons](figs/imsize_comparisons.png)
#
# From the results, we can see that an increase in image resolution from 299X299 to 499X499 will increase the performance marginally at the cost of a longer training duration and slower inference speed.
#
# ---
#
# <details><summary>See the code to generate the graphs</summary>
# <p>
#
# #### Code snippet to generate graphs in this cell
#
# ```python
# import pandas as pd
# from utils_ic.parameter_sweeper import add_value_labels
# %matplotlib inline
#
# df = pd.DataFrame({
#     "accuracy": [.9472, .9394, .9190, .9164, .8366, .8251],
#     "training_duration": [385.3, 218.8, 280.5, 184.9, 272.5, 182.3],
#     "inference_duration": [34.2, 23.2, 27.8, 17.8, 27.6, 17.3],
#     "model": ['resnet50 X 499', 'resnet50 X 299', 'resnet18 X 499', 'resnet18 X 299', 'squeezenet1_1 X 499', 'squeezenet1_1 X 299'],
# }).set_index("model"); df
#
# ax1, ax2, ax3 = df.plot.bar(
#     rot=90, subplots=True, legend=False, figsize=(12, 12)
# )
#
# for i in range(len(df)):
#     if i < len(df)/3:
#         ax1.get_children()[i].set_color('r')
#         ax2.get_children()[i].set_color('r')
#         ax3.get_children()[i].set_color('r')
#     if i >= len(df)/3 and i < 2*len(df)/3:
#         ax1.get_children()[i].set_color('g')
#         ax2.get_children()[i].set_color('g')
#         ax3.get_children()[i].set_color('g')
#     if i >= 2*len(df)/3:
#         ax1.get_children()[i].set_color('b')
#         ax2.get_children()[i].set_color('b')
#         ax3.get_children()[i].set_color('b')
#
# ax1.set_title("Accuracy (%)")
# ax2.set_title("Training Duration (seconds)")
# ax3.set_title("Inference Speed (seconds)")
#
# ax1.set_ylabel("%")
# ax2.set_ylabel("seconds")
# ax3.set_ylabel("seconds")
#
# ax1.set_ylim(top=df["accuracy"].max() * 1.2)
# ax2.set_ylim(top=df["training_duration"].max() * 1.2)
# ax3.set_ylim(top=df["inference_duration"].max() * 1.2)
#
# add_value_labels(ax1, percentage=True)
# add_value_labels(ax2)
# add_value_labels(ax3)
# ```
#
# </p>
# </details>

# ## TLDR <a name="tldr"></a>
#
# Heres the tldr if you didn't have time to read the above.
#
#
# __Architecture__
#
# - Start with __resnet18__. If you're not memory or inference speed constrained, switch to __resnet50__. If you are memory constrained, use __squeezenet1_1__.
#
#
# __Learning rate__
#
# - Set your learning rate to __1e-4__ if you have lots of time to train. Otherwise, __1e-3__ will work pretty well on fewer epochs.
#
#
# __Image size__
#
# - Image size __299__ works pretty well. Increase image size and it'll work a bit better but much slower.
#
# __Other parameters__
#
# - Don't worry about the model's __batch size__, __momentum__, __drop-out__ and __weight decay__. Unless you're really fine-tuning your model, just stick with these [defaults](#methodology-default-parameters).
#

# ---

# # Training our classifier

# Lets first verify our fastai version:

# In[1]:


import fastai

fastai.__version__


# Ensure edits to libraries are loaded and plotting is shown in the notebook.

# In[2]:


get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


# Import fastai. For now, we'll import all (import *) so that we can easily use different utilies provided by the fastai library.

# In[3]:


import sys

sys.path.append("../")
import os
from pathlib import Path
from utils_ic.datasets import Urls, unzip_url
from fastai.vision import *
from fastai.metrics import accuracy


# Now that we've set up our notebook, lets set the hyperparameters based on which model type was selected.

# ## Choosing between two types of models

# For most scenarios, computer vision practitioners want to create one of two types of models:
#
# 1. __Model A__ - A model that performs at its highest possible performance (such as accuracy).
# 1. __Model B__ - A model with a small memory footprint, fast inference speeds, and fast training times.
#
# Based on the findings above, we can get either one of these models by tweaking our paramaters.

# In[4]:


# Only one can be True
MODEL_A = False
MODEL_B = True

# Path to your data
DATA_PATH = unzip_url(Urls.recycle_path, exist_ok=True)


# Make sure that only one is set to True

# In[5]:


assert MODEL_A - MODEL_B != 0


# In[6]:


if MODEL_A:
    ARCHITECTURE = models.resnet50
    IM_SIZE = 499  # you try even higher
    LEARNING_RATE = 1e-4
    EPOCHS = 15

if MODEL_B:
    ARCHITECTURE = models.squeezenet1_1
    IM_SIZE = 299  # you try even lower
    LEARNING_RATE = 1e-3
    EPOCHS = 5


# ## Training
#
# We'll now re-apply the same steps we did in the [training introduction](01_training_introduction.ipynb) notebook here.

# Load our data.

# In[7]:


data = (
    ImageList.from_folder(Path(DATA_PATH))
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_from_folder()
    .transform(size=IM_SIZE)
    .databunch(bs=16)
    .normalize(imagenet_stats)
)


# Create our learner.

# In[8]:


learn = cnn_learner(data, ARCHITECTURE, metrics=accuracy)


# Train the last layer for a few epochs.

# In[9]:


learn.fit_one_cycle(2, LEARNING_RATE)


# Unfreeze the layers

# In[10]:


learn.unfreeze()


# Fine tune the network for the remaining epochs.

# In[11]:


learn.fit_one_cycle(EPOCHS, LEARNING_RATE)


# ## Evaluation

# In this section, we test our model on the following characteristics:
# - accuracy
# - parameter export size / memory footprint required
# - inference speed
#
# Refer back to the [training introduction](01_training_introduction.ipynb) to learn about other ways to evaluate the model.

# #### Accuracy
# For now, we can just take a look at the final accuracy on the validation set.

# In[12]:


_, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
print(f"Accuracy on validation set: {float(metric)}")


# #### Memory footprint
#
# Export our model and inspect the size of the file.

# In[13]:


model_fn = "model_a" if MODEL_A else "model_b"


# In[14]:


learn.export(f"{model_fn}")


# In[15]:


size_in_mb = os.path.getsize(
    Path(DATA_PATH) / ("model_a" if MODEL_A else "model_b")
) / (1024 * 1024.0)
print(f"'{model_fn}' is {round(size_in_mb, 2)}MB.")


# #### Inference speed
#
# Use the model to inference and time how long it takes.

# In[16]:


im = open_image(f"{(Path(DATA_PATH)/learn.data.classes[0]).ls()[0]}")


# In[17]:


get_ipython().run_cell_magic("timeit", "", "learn.predict(im)")


# Now that we have a good understanding of how different parameters affect the model, we can create more specific models to better fit out needs.
