#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Hard Negative Sampling for Image Classification

# You built an image classification model, evaluated it on a validation set and got a decent accuracy. Now you deploy the model for the real-world scenario. And soon, you may find that the model performs worse than expected.
# 
# This is quite common scenario (and inevitable) when we build a machine learning model because we cannot collect all the possible samples. Your model is supposed to learn the features that describe the target classes the best, but in reality, it learns the best features to classify your dataset. For example, if we have photos of *butterfly* on a flower, the model may learn flower shapes to classify *butterfly*.
# 
# <img src="./media/hard_neg_ex1.jpg" width="300"> | <img src="./media/hard_neg_ex2.jpg" width="300"> 
# ---|---
# Did our model learn a butterfly? | or yellow flowers?
# 
# Hard negative sampling (or hard negative mining) is a useful technique to address this pitfall. It is a way to explicitly create examples for your training set from falsely classified samples. The technique is widely used when you cannot add all the negative samples since (i) training time would get too slow because of too many training samples; and (ii) many of the negative images are trivial for the model and hence the model would not learn anything. Therefore, we try to identify the images which make a difference when added to the training set.
# 
# In this notebook, we train our model on a training set as usual, test the model on un-seen negative examples and see if the model classifies them correctly. If not, we introduce those samples into the training set and re-train the model on them.
# 
# # Overview
# 
# Our goal is to train a classifier which can recognize *fridge obejcts* (`watter_bottle`, `carton`, `can`, and `milk_bottle`), similar to [01_train notebook](./01_training_introduction.ipynb). However, the input image might not even contain any of these objects in the real use-case. Therefore, we also introduce `negative` class.
# 
# <img src="./media/hard_neg.jpg" width="600"/>
# 
# The overall training process is as follows: 
# * First, prepare training set <i>T</i> and negative-sample set <i>U</i>. <i>T</i> may include initial negative samples
# * Next, load a pre-trained ImageNet model
# * And then, mine hard negative samples by following steps as shown in the figure:
#     1. Train the model on <i>T</i>
#     2. Score the model on <i>U</i>
#     3. Identify hard images the model mis-classified, annotate them and add to <i>T</i> so that the model can learn the patterns it confused before.
# * Finally, repeat these steps until we get a good accuracy.

# In[1]:


# Ensure edits to libraries are loaded and plotting is shown in the notebook.
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from functools import partial 
import os
from pathlib import Path
import sys
sys.path.append("../../")
import shutil
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
# fastai
import fastai
from fastai.metrics import accuracy
from fastai.vision import (
    # data-modules
    CategoryList, DatasetType, get_image_files, ImageList, imagenet_stats,
    # model-modules
    cnn_learner, models, ClassificationInterpretation, 
)

from utils_cv.classification.model import (
    IMAGENET_IM_SIZE as IMAGE_SIZE,
    TrainMetricsRecorder,
    get_preds,
)
from utils_cv.classification.plot import plot_pr_roc_curves
from utils_cv.classification.widget import ResultsWidget
from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url
from utils_cv.common.gpu import which_processor
from utils_cv.common.misc import copy_files, set_random_seed
from utils_cv.common.plot import line_graph, show_ims

print(f"Fast.ai version = {fastai.__version__}")
which_processor()


# In[3]:


DATA_PATH     = unzip_url(Urls.fridge_objects_path, exist_ok=True)
NEGATIVE_NUM  = 10  # Number of negative samples to add for each iteration of negative mining
EPOCHS_HEAD   = 4
EPOCHS_BODY   = 12
LEARNING_RATE = 1e-4
BATCH_SIZE    = 16
# Using fast_inference parameters from 02_training_accuracy_vs_speed notebook.
ARCHITECTURE  = models.resnet18
IM_SIZE       = 300 

# Temporary folder to store datasets for hard-negative mining
NEGATIVE_MINING_DATA_DIR = TemporaryDirectory().name


# ## 1. Prepare datasets

# We prepare our dataset in the following way:
# * The initial training set `T` to include *fridge objects* as well as some initial *negative samples*<sup>+</sup>.
# * Negative image set `U`.
# * Validation set `V` to have both *fridge objects* and *negative samples*. We evaluate our model on this set.
# 
# <sub>+ We added `NEGATIVE_NUM` of negative samples to our initial training set. In a real use-case, you may want to include 100 or more images of negative samples.</sub>

# In[4]:


ori_datapath = Path(DATA_PATH)
neg_datapath = Path(unzip_url(Urls.fridge_objects_negatives_path, exist_ok=True))
# We split positive samples into 80% training and 20% validation
data_imlist = (ImageList.from_folder(ori_datapath)
               .split_by_rand_pct(valid_pct=0.2, seed=10)
               .label_from_folder())
# We use 80% of negative images for hard-negative mining (set U) while 20% for validation
neg_data = (ImageList.from_folder(neg_datapath)
            .split_by_rand_pct(valid_pct=0.2, seed=10)
            .label_const()  # We don't use labels for negative data
            .transform(size=IMAGE_SIZE) 
            .databunch(bs=BATCH_SIZE)
            .normalize(imagenet_stats))
# Do not shuffle U when we predict
neg_data.train_dl = neg_data.train_dl.new(shuffle=False) 
neg_data


# In[5]:


datapath = Path(NEGATIVE_MINING_DATA_DIR)/'data'

# Training set T
copy_files(data_imlist.train.items, datapath/'train', infer_subdir=True)
# We include first NEGATIVE_NUM negative images in U (neg_data.train_ds) to our initial training set T
copy_files(neg_data.train_ds.items[:NEGATIVE_NUM], datapath/'train'/'negative')

# Validation set V
copy_files(data_imlist.valid.items, datapath/'valid', infer_subdir=True)
copy_files(neg_data.valid_ds.items, datapath/'valid'/'negative')


# In[6]:


set_random_seed(10)


# In[7]:


data = (ImageList.from_folder(datapath)
        .split_by_folder()
        .label_from_folder()
        .transform(size=IMAGE_SIZE) 
        .databunch(bs=BATCH_SIZE) 
        .normalize(imagenet_stats))
data.show_batch()


# In[8]:


print(f'number of classes: {data.c} = {data.classes}')
print(data.batch_stats)


# ## 2. Prepare a model
# 
# We use *fast inference* setup we demonstrated from [02_training_accuracy_vs_speed notebook](./02_training_accuracy_vs_speed.ipynb). The model is Resnet18 and pre-trained on [ImageNet](http://www.image-net.org/). Regarding the details about training concept, please see [01_training notebook](./01_training_introduction.ipynb).

# In[9]:


learn = cnn_learner(data, ARCHITECTURE, metrics=accuracy)


# In[10]:


learn.fit_one_cycle(EPOCHS_HEAD, LEARNING_RATE)


# In[11]:


# Records train and valid accuracies by using Callback TrainMetricsRecorder
learn.callbacks.append(TrainMetricsRecorder(learn, show_graph=True))
learn.unfreeze()


# In[12]:


# We record train and valid accuracies for later analysis
train_acc = []
valid_acc = []
interpretations = []


# ## 3. Train the model on *T*
# 
# <a id='train'></a>
# 
# From this section to the end, we do training and negative mining. As described in the Overview section, You may need to do repeat the negative mining steps several times to achieve good results.

# In[48]:


# Show the number of repetitions you went through the negative mining
print(f"Ran {len(interpretations)} time(s)")


# In[49]:


learn.fit_one_cycle(EPOCHS_BODY, LEARNING_RATE)


# The following cell shows confusion matrix for the validation set. If you are repeating the negative mining steps, you will see all the confusion matrices from the repetitions.

# In[50]:


interpretations.append(ClassificationInterpretation.from_learner(learn))


# In[51]:


for i, interp in enumerate(interpretations):
    interp.plot_confusion_matrix()


# In[52]:


# Store train and valid accuracy
train_acc.extend(np.array(learn.train_metrics_recorder.train_metrics)[:, 0])
valid_acc.extend(np.array(learn.train_metrics_recorder.valid_metrics)[:, 0])


# In[53]:


line_graph(
    values=(train_acc, valid_acc),
    labels=("Train", "Valid"),
    x_guides=[i*EPOCHS_BODY for i in range(1, len(train_acc)//EPOCHS_BODY + 1)],
    x_name="Epoch",
    y_name="Accuracy",
)


# **If the model performs well enough, we can stop the training / negative sampling here.**
# 
# If not, let's do hard negative sampling.

# ## 4. Score the model on *U* 

# In[42]:


pred_outs = np.array(get_preds(learn, neg_data.train_dl)[0].tolist())
print(f"Prediction results:\n{pred_outs[:10]}\n...")


# ## 5. Hard negative mining

# In[43]:


# Get top-n false classified images (by confidence)
preds = np.argmax(pred_outs, axis=1)
wrong_ids = np.where(preds!=data.classes.index('negative'))[0]
wrong_ids_confs = [(i, pred_outs[i][preds[i]]) for i in wrong_ids]
wrong_ids_confs = sorted(wrong_ids_confs, key=lambda l:l[1], reverse=True)[:NEGATIVE_NUM]


# In[44]:


negative_sample_ids = [w[0] for w in wrong_ids_confs]
negative_sample_labels = [f"Pred: {data.classes[preds[w[0]]]}\nConf: {w[1]:.3f}" for w in wrong_ids_confs]
show_ims(neg_data.train_ds.items[negative_sample_ids], negative_sample_labels, rows=NEGATIVE_NUM//5)


# ## 6. Add hard negative samples to the training set *T*
# 
# We add the hard negative samples into the training set.

# In[45]:


copy_files(neg_data.train_ds.items[negative_sample_ids], datapath/'train'/'negative')


# In[47]:


# Reload the dataset which includes more negative-samples
data = (ImageList.from_folder(datapath)
        .split_by_folder()
        .label_from_folder()
        .transform(size=IMAGE_SIZE) 
        .databunch(bs=BATCH_SIZE) 
        .normalize(imagenet_stats))
print(data.batch_stats)

# Set the dataset to the learner
learn.data = data


# Now, let's go **back** to the "[3. Train the model on T](#train)" and repeat the training and negative mining steps while we have a decent accuracy on `negative` samples

# In[54]:


# Finally, show the number of repetitions you went through the negative mining
print(f"Ran {len(interpretations)} time(s)")


# In[ ]:




