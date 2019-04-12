#!/usr/bin/env python
# coding: utf-8

# # Building Models for Accuracy VS Speed
#
# The goal of this notebook is to understand how to train a model with different parameters to achieve either a highly accurate but slow model, or a model with fast inference speed but with lower accuracy.
#
# As practitioners of computer vision, we want to be able to control what to optimize when building our models. Unless you are building a model for a Kaggle competition, it is unlikely that you can build your model with only its accuracy in mind.
#
# For example, in an IoT setting, where the inferencing device has limited computational capabilities, we need to design our models to have a small memory footprint. In contrast, medical situations often require the highest possible accuracy because the cost of mis-classification could impact the well-being of a patient. In this scenario, the accuracy of the model can not be compromised.
#
# We have conducted various experiments on multiple diverse datasets to find parameters which work well on a wide variety of settings, for e.g. high accuracy or fast inference. In this notebook, we provide these parameters, so that your initial models can be trained without any parameter tuning. For most datasets, these parameters are close to optimal, so there won't need to change them much. In the second part of the notebook, we will give guidelines as to what parameters could be fine-tuned and how they impact the model, and which parameters typically do not have a big influence
#
# It is recommended that you first train your model with the default parameters, evaluate the results, and then only as needed, try fine tuning parameters to achieve better results.

# ## Table of Contents:
# * [Training a High Accuracy or a Fast Inference Speed Classifier ](#model)
#   * [Choosing between two types of models](#choosing)
#   * [Training](#training)
#   * [Evaluation](#evaluation)
# * [Fine tuning our models](#finetuning)
#   * [DNN Architecture](#dnn)
#   * [Key Parameters](#key-parameters)
#   * [Other Parameters](#other-parameters)
#   * [Testing Parameters](#testing-parameters)
# * [Appendix](#appendix)
#   * [Learning Rate](#appendix-learning-rate)
#   * [Image Size](#appendix-imsize)
#   * [How we got good parameters](#appendix-good-parameters)

# ## Training a High Accuracy or a Fast Inference Speed Classifier <a name="model"></a>

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

# ### Choosing between two types of models <a name="choosing"></a>

# For most scenarios, computer vision practitioners want to create a high accuracy model, a fast-inference model or a small size model. Set your `MODEL_TYPE` variable to one of the following: `"high_accuracy"`, `"fast_inference"`, or `"small_size"`.
#
# For this notebook, we'll be using the FridgeObjects dataset as we did in the [previous notebook](01_training_introduction.ipynb). You can replace the `DATA_PATH` variable with your own data by passing its path.
#
# When choosing your batch size, its worth noting that even mid-level GPUs run out of memory when training a deeper resnet models at larger image resolutions. If you get an _out of memory_ error, try reducing the batch size by a factor of 2, and try again.

# In[4]:


# Choose between "high_accuracy", "fast_inference", or "small_size"
MODEL_TYPE = "fast_inference"

# Path to your data
DATA_PATH = unzip_url(Urls.fridge_objects_path, exist_ok=True)

# Epochs to train for
EPOCHS_HEAD = 4
EPOCHS_BODY = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 16


# Make sure that only one is set to True

# In[5]:


assert MODEL_TYPE in ["high_accuracy", "fast_inference", "small_size"]


# Set parameters based on your selected model.

# In[6]:


if MODEL_TYPE == "high_acccuracy":
    ARCHITECTURE = models.resnet50
    IM_SIZE = 500

if MODEL_TYPE == "fast_inference":
    ARCHITECTURE = models.resnet18
    IM_SIZE = 300

if MODEL_TYPE == "small_size":
    ARCHITECTURE = models.squeezenet1_1
    IM_SIZE = 300


# ### Training <a name="training"></a>
#
# We'll now re-apply the same steps we did in the [training introduction](01_training_introduction.ipynb) notebook here.

# Load our data.

# In[7]:


data = (
    ImageList.from_folder(Path(DATA_PATH))
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_from_folder()
    .transform(tfms=get_transforms(), size=IM_SIZE)
    .databunch(bs=16)
    .normalize(imagenet_stats)
)


# Create our learner.

# In[8]:


learn = cnn_learner(data, ARCHITECTURE, metrics=accuracy)


# Train the last layer for a few epochs.

# In[9]:


learn.fit_one_cycle(EPOCHS_HEAD, LEARNING_RATE)


# Unfreeze the layers

# In[10]:


learn.unfreeze()


# Fine tune the network for the remaining epochs.

# In[11]:


learn.fit_one_cycle(EPOCHS_BODY, LEARNING_RATE)


# ### Evaluation <a name="evaluation"></a>

# In this section, we test our model on the following characteristics:
# - accuracy
# - inference speed
# - parameter export size / memory footprint required
#
#
# Refer back to the [training introduction](01_training_introduction.ipynb) to learn about other ways to evaluate the model.

# #### Accuracy
# To keep things simple, we just a look at the final accuracy on the validation set.

# In[12]:


_, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
print(f"Accuracy on validation set: {float(metric)}")


# #### Inference speed
#
# Use the model to inference and time how long it takes.

# In[13]:


im = open_image(f"{(Path(DATA_PATH)/learn.data.classes[0]).ls()[0]}")


# In[14]:


get_ipython().run_cell_magic("timeit", "", "learn.predict(im)")


# #### Memory footprint
#
# Export our model and inspect the size of the file.

# In[15]:


learn.export(f"{MODEL_TYPE}")


# In[16]:


size_in_mb = os.path.getsize(Path(DATA_PATH) / MODEL_TYPE) / (1024 * 1024.0)
print(f"'{MODEL_TYPE}' is {round(size_in_mb, 2)}MB.")


# ---

# ## Fine tuning parameters <a name="finetuning"></a>
#
# If you use the parameters provided in the repo along with the defaults that Fastai provides, you can get good results across a wide variety of datasets. However, as is true for most machine learning projects, getting the best possible results for a new dataset requires tuning the parameters that you use. The following section provides guidelines on how to optimize for accuracy, inference speed, or model size on a given dataset. We'll go through the parameters that will make the largest impact on your model as well as the parameters that may not be worth tweaking.
#
# Generally speaking, models for image classification comes with a trade-off between training time versus model accuracy. The four parameters that most affect this trade-off are the DNN architecture, image resolution, learning rate, and number of epochs. DNN architecture and image resolution will additionally affect the model's inference time and memory footprint. As a rule of thumb, deeper networks with high image resolution will achieve higher accuracy at the cost of large model sizes and low training and inference speeds. Shallow networks with low image resolution will result in models with fast inference speed, fast training speeds and low model sizes at the cost of the model's accuracy.

# ### DNN Architectures <a name="dnn"></a>
#
# When choosing at an architecture, we want to make sure it fits our requirements for accuracy, memory footprint, inference speed and training speeds. Some DNNs have hundreds of layers and end up having quite a large memory footprint with millions of parameters to tune, while others are compact and small enough to fit onto memory limited edge devices.
#
# Lets take a __squeezenet1_1__ model, a __resnet18__ model and __resnet50__ model and compare the differences based on our experiment that is based of a diverse set of 6 different datasets. (More about the datasets in the appendix below)
#
# ![architecture_comparisons](figs/architecture_comparisons.png)
#
# As you can see from the graphs above, there is a clear trade-off when deciding between the models.
#
# In terms of accuracy, __resnet50__ out-performs the rest, but it also suffers from having the highest memory footprint, and the longest training and inference times. On the other end of the spectrum, __squeezenet1_1__ performs the worst in terms fo accuracy, but has by far the smallest memory footprint.
#
# Generally speaking, given enough data, the deeper DNN and the higher the image resolution, the higher the accuracy you'll be able to achieve with your model.
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

# ### Key Parameters <a name="key-parameters"></a>
# This section examines some of the key parameters when training a deep learning model for image classification. The table below shows default parameters we recommend using.
#
# | Parameter | Default Value |
# | --- | --- |
# | Learning Rate | 1e-4 |
# | Epochs | 15 |
# | Batch Size | 16 |
# | Image Size | 300 X 300 |
#
# __Learning rate__
#
# Learning rate step size is used when optimizing your model with gradient descent and tends to be one of the most important parameters to set when training your model. If your learning rate is set too low, training will progress very slowly since we're only making tiny updates to the weights in your network. However, if your learning rate is too high, it can cause undesirable divergent behavior in your loss function. Generally speaking, choosing a learning rate of 1e-4 was shown to work pretty well for most datasets. If you want to reduce training time (by training for fewer epochs), you can try setting the learning rate to 5e-3, but if you notice a spike in the training or validation loss, you may want to try reducing your learning rate.
#
# You can learn more about learning rate in the [appendix below](#appendix-learning-rate).
#
# __Epochs__
#
# When it comes to choosing the number of epochs, a common question is - _Won't too many epochs will cause overfitting_? It turns out that the accuracy on the test set typically does not get worse, even if training for too many epochs. Unless your are working with small datasets, using around 15 epochs tends to work pretty well in most cases.
#
#
# __Batch Size__
#
# Batch size is the number of training samples you use in order to make one update to the model parameters. A batch size of 16 or 32 works well for most cases. The higher the batch size, the faster training will be, but at the expense of an increased DNN memory consumption. Depending on your dataset and the GPU you have, you can start with a batch size of 32, and move down to 16 if your GPU doesn't have enough memory. After a certain increase in batch size, improvments to training speed become marginal, hence we found 16 (or 32) to be a good trade-off between training speed and memory consumption.If you reduce the batch size, you may also have to reduce the learning rate.
#
# __Image size__
#
# The default image size is __300 X 300__ pixels. Using higher image resolution of, for example, __500 X 500__ or even higher, can improve the accuracy of the model but at the cost of longer training and inference times.
#
# You can learn more about the impact of image resolution in the [appendix below](#appendix-imsize).
#

# ### Other Parameters <a name="other-parameters"></a>
#
# In this section, we examine some of the other common hyperparameters when dealing with DNNs. The key take-away is that that exact value of these parameters do not have a big impact on the model's performance, training/inference speed, or memory footprint.
#
# | Parameter | Good Default Value |
# | --- | --- |
# | Dropout | 0.5 or (0.5 on the final layer and 0.25 on all previous layers) |
# | Weight Decay | 0.01 |
# | Momentum | 0.9 or (min=0.85 and max=0.95 when using cyclical momentum) |
#
# __Dropout__
#
# Dropout is a way to discard activations at random when training your model. It is a way to keep the model from over-fitting on the training data. In Fastai, dropout is by default set to 0.5 on the final layer, and 0.25 on all previous layer. Unless there is clear evidence of over-fitting, drop out tends to work well at this default so there is no need to change it much.
#
# __Weight decay (L2 regularization)__
#
# Weight decay is a regularization term applied when minimizing the network's loss. We can think of it as a penalty applied to the weights after an update. This will help prevent the weights from growing too big. In Fastai, the default weight decay is 0.1, which is what we should leave it at.
#
# __Momentum__
#
# Momentum is a way to reach convergence faster when training our model. It is a way to incorporate a weighted average of the most recent updates to the current update. Fastai implements cyclical momentum when calling `fit_one_cycle()`, so the momentum will fluctuate over the course of the training cycle, hence we need a min and max value for momentum.
#
# When using `fit_one_cycle()`, the default value of max=0.95 and min=0.85 is shown to work well. If using `fit()`, the default value of 0.9 has been shown to work well. These defaults provided by Fastai represent a good trade-off between training speed and the ability of the model to converge to a good solution

# ### Testing Parameters <a name="testing-parameters"></a>
# If you want to fine tune parameters and test different parameters, you can use the ParameterSweeper module the find the best parameter. See the [exploring hyperparameters notebook](./11_exploring_hyperparameters.ipynb) for more information.

# ---

# # Appendix <a name="appendix"></a>

# ### Learning Rate <a name="appendix-learning-rate"></a>
#
# One way to mitigate against a low learning rate is to make sure that you're training for many epochs. But this can take a long time.
#
# So, to efficiently build a model, we need to make sure that our learning rate is in the correct range so that we can train for as few epochs as possible. To find a good default learning rate, we've tested various learning rates on 6 different datasets, training the full network for 3 or 15 epochs.
#
# ![lr_comparisons](figs/lr_comparisons.png)
#
# <details><summary><em>Understanding the diagram</em></summary>
# <p>
#
# > In the figure on the left which shows the results of the different learning rates on different datasets at 15 epochs, we can see that a learning rate of 1e-4 does the best overall. But this may not be the case for every dataset. If you look carefully, there is a pretty significant variance between the datasets and it may be possible that a learning rate of 1-e3 works better than a learning rate of 1e-4 for some datasets. In the figure on the right, both 1e-4 and 1e-3 seem to work well. At 15 epochs, the results of 1e-4 are only slightly better than that of 1e-3. However, at 3 epochs, a learning rate of 1e-3 out performs the learning rate at 1e-4. This makes sense since we're limiting the training to only 3 epochs, the model that can update its weights more quickly will perform better. As a result, we may learn towards using higher learning rates (such as 1e-3) if we want to minimize the training time, and lower learning rates (such as 1e-4) if training time is not constrained.
#
# </p>
# </details>
#
# In both figures, we can see that a learning rate of 1e-3 and 1e-4 tends to work the best across the different datasets and the two settings for epochs. We observe that training using only 3 epochs gives inferior results compared to 15 epochs. Generally speaking, choosing a learning rate of 5e-3 (the mean of 1e-3 and 1e-4) was shown to work pretty well for most datasets. However, for some datasets, a learning rate of 5-e3 will cause the training to diverge. In those cases, try a lower epoch, like 1e-4.
#
# Fastai has implemented [one cycle policy with cyclical momentum](https://arxiv.org/abs/1803.09820) which requires a maximum learning rate since the learning rate will shift up and down over its training duration. Instead of calling `fit()`, we simply call `fit_one_cycle()`.
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

# ### Image Resolution <a name="appendix-imsize"></a>
#
# A model's input image resolution tends to affect its accuracy. Usually, convolutional neural networks are able to take advantage of higher resolution images. This is especially true is the object-of-interest is small in the image.
#
# But how does it impact some of the other aspects of the model?
#
# It turns out that the image size doesn't affect the model's memory footprint, but it has a huge effect on GPU memory. Image size also has a direct impact on training and inference speeds. An increase in image size will result in slower inference speeds.
#
# ![imsize_comparisons](figs/imsize_comparisons.png)
#
# From the results, we can see that an increase in image resolution from __300 X 300__ to __500 X 500__ will increase the performance marginally at the cost of a longer training duration and slower inference speed.
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

# ### How we found good default parameters <a name="#appendix-good-parameters"></a>
#
# To explore the charactistics of a model, we - the computer vision repo team - have conducted various experiments to explore the impact of different hyperparameters on a model's _accuracy_, _training duration_, _inference speed_, and _memory footprint_. In this notebook, we used the results of our experiments to give us concrete evidence when it comes to understanding which parameters work and which dont.
#
# #### Datasets <a name="datasets"></a>
#
# For our experiments, we relied on a set of six different classification datasets. When selecting these datasets, we wanted to have a variety of image types with different amounts of data and number of classes.
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
# #### Model Characteristics <a name="model-characteristics"></a>
#
# In our experiment, we look at these characteristics to evaluate the impact of various paramters. Here is how we calculated each of the following metrics:
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
