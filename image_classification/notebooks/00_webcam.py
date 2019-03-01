#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # WebCam Image Classification Quickstart Notebook
# 
# <br>
# 
# Image classification is a classical problem in computer vision that determining whether or not the image data contains some specific object, feature, or activity. It is regarded as a mature research area
# and currently the best models are based on [convolutional neural networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network). Such models with weights trained on millions of images and hundreds of object classes in [ImageNet dataset](http://www.image-net.org/) are available from major deep neural network frameworks such as [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/), [fast.ai](https://docs.fast.ai/vision.models.html#Computer-Vision-models-zoo), [Keras](https://keras.io/applications/), [PyTorch](https://pytorch.org/docs/stable/torchvision/models.html), and [TensorFlow](https://tfhub.dev/s?module-type=image-classification).
# 
# 
# This notebook shows a simple example of how to load pretrained mobel and run it on a webcam stream. Here, we use [ResNet](https://arxiv.org/abs/1512.03385) model by utilizing `fastai.vision` package.
# 
# > For more details about image classification tasks including transfer-learning (aka fine tuning), please see our [training introduction notebook](01_training_introduction.ipynb).

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sys
sys.path.append("../")

import io, logging, shutil, time
from utils_ic.datasets import imagenet_labels

import fastai
from fastai.vision import *
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlInit

from ipywebrtc import CameraStream, ImageRecorder
import ipywidgets as widgets


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

print("Fast.ai:", fastai.__version__)

nvmlInit()
try: 
    handle = nvmlDeviceGetHandleByIndex(0)
    print("Device 0:", nvmlDeviceGetName(handle))
except NVMLError as error:
    print(error)


# In[3]:


IMAGE_SIZE = 224


# ## 1. Load Pretrained Model
# 
# We use ResNet18 which is a relatively small and fast compare to other CNNs models. The [reported error rate](https://pytorch-zh.readthedocs.io/en/latest/torchvision/models.html) of the model on ImageNet is 30.24% for top-1 and 10.92% for top-5<sup>*</sup>.
# 
# The pretrained model expects input images normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225], which is defined in `fastai.vision.imagenet_stats`.
# 
# The output of the model is the probability distribution of the classes in ImageNet. To convert them into human-readable labels, we utilize the label json file used from [Keras](https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py).
# 
# > \* top-n: *n* labels considered most probable by the mode

# In[4]:


labels = imagenet_labels()
print("Num labels =", len(labels))
print(", ".join(labels[:5]), "...")


# In[5]:


empty_data = ImageDataBunch.single_from_classes(
    "", classes=labels, size=IMAGE_SIZE
).normalize(imagenet_stats)

learn = Learner(empty_data, models.resnet18(pretrained=True))


# ## 2. Classify Images
# 
# 
# ### 2.1 Image file
# We use a sample image from `COCO_TINY` in `fastai.vision` just because it is a very small size and easy to download by calling a simple function, `untar_data(URLs.COCO_TINY)`.
# 
# > Original [COCO dataset](http://cocodataset.org/) is a large-scale object detection, segmentation, and captioning dataset.

# In[6]:


path = untar_data(URLs.COCO_TINY)
im = open_image(path/"train"/"000000564902.jpg", convert_mode='RGB')
im


# In[7]:


_, ind, _ = learn.predict(im)
print(labels[ind])


# ### 2.2 WebCam Stream
# 
# We use `ipywebrtc` to start a webcam and get the video stream to the notebook's widget. For details about `ipywebrtc`, see [this link](https://ipywebrtc.readthedocs.io/en/latest/). 

# In[8]:


run_model = True

# Webcam
cam = CameraStream(
    constraints={
        'facing_mode': 'user',
        'audio': False,
        'video': { 'width': IMAGE_SIZE, 'height': IMAGE_SIZE }
    }
)
# Image recorder for taking a snapshot
im_recorder = ImageRecorder(stream=cam, layout=widgets.Layout(margin='0 0 0 50px'))
# Text label widget to show our classification results
label = widgets.Label("result label") 

# For every snapshot, we run the pretrained model
def classify_frame(_):
    im_recorder.recording = True
    
    try:
        im = open_image(io.BytesIO(im_recorder.image.value), convert_mode='RGB')
        _, ind, _ = learn.predict(im)
        label.value = labels[ind]
    except OSError:
        pass

    if run_model:
        im_recorder.recording = True
        
im_recorder.image.observe(classify_frame, 'value')


# In[9]:


# Show widgets
widgets.HBox([cam, im_recorder, label])


# Now, click the **capture button** of the image recorder widget to start classification. Labels show the most probable class predicted by the model for an image snapshot.

# <img src="../docs/media/webcam.png" style="width: 400px;"/>
# 
# The results maybe not very accurate because the model was not trained on the webcam images. To make the model perform better on new image classification problem, we usually fine-tune the model with new data. Examples about this transfer learning can be found from our [training introduction notebook](01_training_introduction.ipynb).

# In[10]:


stop_process = True
widgets.Widget.close_all()


# In[ ]:


# Cleanup
shutil.rmtree(path, ignore_errors=True)


# In[ ]:




