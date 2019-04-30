#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Hard Negative Sampling for Image Classification

# You built an image classification model, evaluated it on a validation set and got a decent accuracy. Now you deploy the model for the real-world scenario. And soon, you'll find that the model underperforms than expected.
# 
# This is quite common scenario (and inevitable) when we build a machine learning model because we cannot collect all the possible samples. Your model is supposed to learn the features that describe the target classes the best, but in reality, it learns the best features **to classify your dataset**. For example, if we have *carton* photos with white background only, the model may learn the background color to classify *carton* objects.
# 
# Hard negative sampling (or hard negative mining) is a useful technique to address this pitfall, that is when you find falsely classified samples, explicitly create examples from them and add to your training set. The technique is widely used in obejct detection scenario where you cannot collect all the negative (non-target object) samples. This is why the technique is called hard 'negative' sampling.
# 
# This concept can be applied to image classification problems. In this notebook, we train our model on a training set as usual, test the model on un-seen data and see if the model performs well. If not, we introduce the misclassified (hard) samples into the training set and re-train the model with it.
# 
# Training workflow is as follows:
# 1. Prepare training set `T` and unlabeled set `U`
# 2. Load a pre-trained ImageNet model
# 3. Train the model on `T`
# 4. Score the model on `U`
# 5. Identify hard images in `U`
# 6. Annotate the hard samples and add to `T`
# 7. Repeat step 3-6 if needed
# 
# Let's get started.

# In[1]:


# Ensure edits to libraries are loaded and plotting is shown in the notebook.
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


import sys
sys.path.append("../../")
from functools import partial 

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# fastai
import fastai
from fastai.vision import (
    # data-modules
    CategoryList, DatasetType, get_image_files, ImageList, imagenet_stats, untar_data, URLs, 
    # model-modules
    cnn_learner, models, ClassificationInterpretation, 
)
from fastai.metrics import accuracy

from utils_cv.classification.model import (
    IMAGENET_IM_SIZE as IMAGE_SIZE,
    TrainMetricsRecorder
)
from utils_cv.classification.plot import plot_pr_roc_curves
from utils_cv.classification.widget import ResultsWidget
from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url
from utils_cv.common.gpu import which_processor
from utils_cv.common.image import show_im_files
from utils_cv.common.misc import set_random_seed

print(f"Fast.ai version = {fastai.__version__}")
which_processor()


# In[3]:


EPOCHS        = 5
LEARNING_RATE = 1e-4
BATCH_SIZE    = 16
ARCHITECTURE  = models.resnet50
SEED          = 10


# ## 1. Prepare datasets

# To demonstrate the hard negative sampling usecase, we introduce one more class into our *fridge obejct* classification problem: the class *pet*. For that, we use [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/).
# 
# So, our end-goal is to classify an image into one of following classes: `watter_bottle`, `carton`, `can`, `milk_bottle`, and `pet`.
# 
# First, let's see how the two datasets look like.

# In[4]:


fridge_datapath = Path(unzip_url(Urls.fridge_objects_path, exist_ok=True))
pet_datapath = Path(untar_data(URLs.PETS))/'images'


# In[5]:


fridge_object_examples = [
    get_image_files(fridge_datapath/'water_bottle')[0],
    get_image_files(fridge_datapath/'carton')[0],
    get_image_files(fridge_datapath/'can')[0],
    get_image_files(fridge_datapath/'milk_bottle')[0],
]
pet_examples = get_image_files(pet_datapath)[:4]

show_im_files(fridge_object_examples, ['water_bottle', 'carton', 'can', 'milk_bottle'])
show_im_files(pet_examples)


# Now, we prepare three datasets out of those two as follows:
# * The initial training set `T` to include *fridge objects* as well as **some of** *pet* images.
# * Unlabeled set `U`. Here, we include only *pets* for simplicity (so that we don't need to browse and annotate manually). 
# * Validation set `V` to have both *fridge objects* and *pets*. We evaluate our model on this set.

# In[6]:


fridge_objects = (ImageList.from_folder(fridge_datapath)
                  .split_by_rand_pct(valid_pct=0.2, seed=SEED)
                  .label_from_folder())
pets = (ImageList.from_folder(pet_datapath)
        .split_by_rand_pct(valid_pct=0.2, seed=SEED))


# In[7]:


import os
import shutil
from tempfile import TemporaryDirectory
tmpdir = TemporaryDirectory()
data_path = Path(tmpdir.name)/'data'


def copy_files(files, dst):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy(f, dst)

# Training set T
copy_files(fridge_objects.train.items, data_path/'train')
copy_files(pets.train.items[:10], data_path/'train')  # add 10 pet images to our initial training set.

# Validation set V
copy_files(fridge_objects.valid.items, data_path/'valid')
copy_files(pets.valid.items, data_path/'valid')

# Test set, in our case, it is U (unlabeled data)
copy_files(pets.train.items[10:], data_path/'test')


# In[8]:


# label function
y_dict = {
    os.path.basename(n): str(fridge_objects.train.y[i]) for i, n in enumerate(fridge_objects.train.items)
}
y_dict.update({
    os.path.basename(n): str(fridge_objects.valid.y[i]) for i, n in enumerate(fridge_objects.valid.items)
})

def get_y_fn(x):
    x = os.path.basename(x)
    return y_dict.get(x) if x in y_dict else 'pet'


# In[9]:


set_random_seed(SEED)
data = (ImageList.from_folder(data_path)
        .split_by_folder()
        .label_from_func(get_y_fn)
        .add_test_folder()
        .transform(size=IMAGE_SIZE) 
        .databunch(bs=BATCH_SIZE) 
        .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(15,11))


# In[10]:


print(f'number of classes: {data.c} = {data.classes}')
print(data.batch_stats)


# > Note, test set doesn't have labels (`EmptyLabelList`)

# ## 2. Load a pre-trained ImageNet model
# 
# We use [ResNet50](https://arxiv.org/abs/1512.03385) model pre-trained on [ImageNet](http://www.image-net.org/), same as [01_training_introduction notebook](./01_training_introduction.ipynb).

# In[11]:


learn = cnn_learner(
    data,
    ARCHITECTURE,
    metrics=[accuracy],
    callback_fns=[partial(TrainMetricsRecorder, show_graph=True)]
)


# ## 3. Train the model on *T*

# In[12]:


# We are training all the layers
learn.unfreeze()


# In[13]:


learn.fit(EPOCHS, LEARNING_RATE)


# In[14]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[15]:


interp.most_confused()


# As you can see, our model confuses `pet` with `carton`.

# ## 4. Score the model on *U* 

# In[16]:


pred_outs = learn.get_preds(ds_type=DatasetType.Test)  # Note, 'Test' is our unlabeled-set U
pred_outs = np.array(pred_outs[0].tolist())  # Convert Tensor to np array


# ## 5. Hard negative sampling

# In[17]:


preds = np.argmax(pred_outs, axis=1)
wrong_pred_ids = np.where(preds!=3)[0]  # We already know the dataset U only includes 'pet' == 3


# In[19]:


wrong_labels = [data.classes[i] for i in preds[wrong_pred_ids][:10]]
show_im_files(data.test_ds.items[wrong_pred_ids[:10]], wrong_labels, rows=2)


# Our model classified cute pets as `carton`, `water_bottle`, and `mile_bottle`. Let's fix this.

# ## 6. Add hard images to training set *T*
# 
# We add the hard samples into the training set and re-train the model.

# In[20]:


copy_files(data.test_ds.items[wrong_pred_ids], data_path/'train')


# In[21]:


new_data = (ImageList.from_folder(data_path)
    .split_by_folder()
    .label_from_func(get_y_fn)
    .transform(size=IMAGE_SIZE) 
    .databunch(bs=BATCH_SIZE) 
    .normalize(imagenet_stats))


# In[22]:


learn.data = new_data
learn.fit(EPOCHS, LEARNING_RATE)


# As you can see, we have low training-accuracy at the beginning even we re-use the model we already trained. This is because of the newly added hard examples.
# 
# Finally, let's see how the model learned from the new data.

# In[32]:


interp_new = ClassificationInterpretation.from_learner(learn)
interp_new.plot_confusion_matrix()


# Below is the previous confusion matrix we showed earlier (the result before we re-trained the model on the hard-samples)

# In[33]:


interp.plot_confusion_matrix()


# After the model learned from hard samples, it classifies `pet` and `carton` much better! 
# 

# In[ ]:




