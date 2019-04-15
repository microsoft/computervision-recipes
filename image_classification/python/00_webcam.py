#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # WebCam Image Classification Quickstart Notebook
# 
# <br>
# 
# Image classification is a classical problem in computer vision that of determining whether or not the image data contains some specific object, feature, or activity. It is regarded as a mature research area
# and currently the best models are based on [convolutional neural networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network). Such models with weights trained on millions of images and hundreds of object classes in [ImageNet dataset](http://www.image-net.org/) are available from major deep neural network frameworks such as [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/), [fast.ai](https://docs.fast.ai/vision.models.html#Computer-Vision-models-zoo), [Keras](https://keras.io/applications/), [PyTorch](https://pytorch.org/docs/stable/torchvision/models.html), and [TensorFlow](https://tfhub.dev/s?module-type=image-classification).
# 
# 
# This notebook shows a simple example of how to load pretrained model and run it on a webcam stream. Here, we use [ResNet](https://arxiv.org/abs/1512.03385) model by utilizing `fastai.vision` package.
# 
# > For more details about image classification tasks including transfer-learning (aka fine tuning), please see our [training introduction notebook](01_training_introduction.ipynb).

# ### Prerequisite for Webcam example 
# You will need to run this notebook on **a machine with a webcam**. We use `ipywebrtc` module to show the webcam widget<sup>*</sup> on the notebook. Currently, the widgets work on **Chrome** and **Firefox**. For more details about the widget, please visit `ipywebrtc` [github](https://github.com/maartenbreddels/ipywebrtc) or [doc](https://ipywebrtc.readthedocs.io/en/latest/).

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sys
sys.path.append("../")
import io
import os
import time
import urllib.request

import fastai
from fastai.vision import models, open_image
from ipywebrtc import CameraStream, ImageRecorder
from ipywidgets import HBox, Label, Layout, Widget

from utils_ic.common import data_path
from utils_ic.constants import IMAGENET_IM_SIZE
from utils_ic.datasets import imagenet_labels
from utils_ic.gpu_utils import which_processor
from utils_ic.imagenet_models import model_to_learner


print(f"Fast.ai: {fastai.__version__}")
which_processor()


# ## 1. Load Pretrained Model
# 
# We use pretrained<sup>*</sup> ResNet18 which is a relatively small and fast among the well-known CNNs architectures. The [reported error rate](https://pytorch.org/docs/stable/torchvision/models.html) of the model on ImageNet is 30.24% for top-1 and 10.92% for top-5 (top five labels considered most probable by the model).
# 
# The model expects input RGB-images to be loaded into a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225], which is defined in [`fastai.vision.imagenet_stats`](https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#L78).
# 
# The output of the model is the probability distribution of the classes in ImageNet. To convert them into human-readable labels, we utilize the label json file used from [Keras](https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py).
# 
# > \* The model is pretrained on ImageNet. Note you can load your own model by using `learn = load_learner(path)` and use it. To learn more about model-export and load, see fastai [doc](https://docs.fast.ai/basic_train.html#Deploying-your-model)).
# 

# In[3]:


labels = imagenet_labels()
print(f"Num labels = {len(labels)}")
print(f"{', '.join(labels[:5])}, ...")


# Note, Fastai's **[Learner](https://docs.fast.ai/basic_train.html#Learner)** is the trainer for model using data to minimize loss function with optimizer. We follow Fastai's naming convention, *'learn'*, for a `Learner` object variable.

# In[4]:


# Convert a pretrained imagenet model into Learner for prediction. 
# You can load an exported model by learn = load_learner(path) as well.
learn = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)


# ## 2. Classify Images
# 
# ### 2.1 Image file
# First, we prepare a coffee mug image to show an example of how to score a single image by using the model.

# In[5]:


# Download an example image
IM_URL = "https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg"
urllib.request.urlretrieve(IM_URL, os.path.join(data_path(), "example.jpg"))

im = open_image(os.path.join(data_path(), "example.jpg"), convert_mode='RGB')
im


# In[6]:


start_time = time.time()

# Use the model to predict the class label
_, ind, prob = learn.predict(im)
print(f"Predicted label: {labels[ind]} (conf = {prob[ind]:.2f})")

# Show prediction time. Note the first prediction usually takes longer because of the model loading
print(f"Took {time.time()-start_time} sec")


# ### 2.2 WebCam Stream
# 
# Now, let's use WebCam stream for image classification. We use `ipywebrtc` to start a webcam and get the video stream to the notebook's widget.

# In[7]:


# Webcam
w_cam = CameraStream(
    constraints={
        'facing_mode': 'user',
        'audio': False,
        'video': { 'width': IMAGENET_IM_SIZE, 'height': IMAGENET_IM_SIZE }
    },
    layout=Layout(width=f'{IMAGENET_IM_SIZE}px')
)
# Image recorder for taking a snapshot
w_imrecorder = ImageRecorder(stream=w_cam, layout=Layout(padding='0 0 0 50px'))
# Label widget to show our classification results
w_label = Label(layout=Layout(padding='0 0 0 50px'))

def classify_frame(_):
    """ Classify an image snapshot by using a pretrained model
    """
    # Once capturing started, remove the capture widget since we don't need it anymore
    if w_imrecorder.layout.display != 'none':
        w_imrecorder.layout.display = 'none'
        
    try:
        im = open_image(io.BytesIO(w_imrecorder.image.value), convert_mode='RGB')
        _, ind, prob = learn.predict(im)
        # Show result label and confidence
        w_label.value = f"{labels[ind]} ({prob[ind]:.2f})"
    except OSError:
        # If im_recorder doesn't have valid image data, skip it. 
        pass
    
    # Taking the next snapshot programmatically
    w_imrecorder.recording = True

# Register classify_frame as a callback. Will be called whenever image.value changes. 
w_imrecorder.image.observe(classify_frame, 'value')


# In[8]:


# Show widgets
HBox([w_cam, w_imrecorder, w_label])


# Now, click the **capture button** of the image recorder widget to start classification. Labels show the most probable class along with the confidence predicted by the model for an image snapshot.
# 
# <center>
# <img src="https://cvbp.blob.core.windows.net/public/images/cvbp_webcam.png" style="width: 400px;"/>
# <i>Webcam image classification example</i>
# </center>

# In this notebook, we have shown a quickstart example of using a pretrained model to classify images. The model, however, is not able to predict the object labels that are not part of ImageNet. From our [training introduction notebook](01_training_introduction.ipynb), you can find how to fine-tune the model to address such problems.

# In[9]:


# Stop the model and webcam
Widget.close_all()


# In[ ]:




