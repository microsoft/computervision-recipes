# Object Detection

## Frequently asked questions

This document tries to answer frequent questions around object detection. For generic ML questions, such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see also the image classification [FAQ](https://github.com/microsoft/ComputerVision/blob/master/classification/FAQ.md).

* General
  * [How does the technology work?](#how-does-the-technology-work)
  * [Which problems can be solved using object detection?](#which-problems-can-be-solved-using-object-detection)

* Technology
  * [R-CNN object detection approaches](#r-cnn-object-detection-approaches)
  * [Intersection-over-Union overlap metric](intersection-over-union-overlap-metric)
  * [Non-maxima suppression](#non-maxima-suppression)
  * [Mean Average Precision](#mean-average-precision)

## General

### How does the technology work?
State-of-the-art object detection methods, such as used in this repository, are based on Convolutional Neural Networks (CNN), a special group of Deep Learning (DL) approaches shown to work well on image data.

One advantage of CNNs is the ability to reuse a CNN trained on millions of images (typically using the [ImageNet](http://image-net.org/index) data set). Such a pre-trained model can be fine-tuned to solve a custom Computer Vision problem given only a small amount of images (in the 100s). This is further explained, and code examples provided, in the [classification](https://github.com/microsoft/ComputerVision/blob/master/classification/) folder of this repository.

Object Detection methods use a pre-trained ImageNet model as backbone.


### Which problems can be solved using object detection?
TODO
Image classification can be used if the object-of-interest is relatively large in the image (more than 20% image width/height). If the object is smaller, or if the location of the object is required, then _object detection_ methods should be used instead.


## Technology


### R-CNN Object Detection Approaches
R-CNNs for Object Detection were first presented in 2014 by [Ross Girshick et al.](http://arxiv.org/abs/1311.2524), and shown to outperform previous state-of-the-art approaches on one of the major object recognition challenges in the field: [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/). The main drawback of the approach was its slow inference speed. Since then, two follow-up papers were published which introduced significant speed improvements: [Fast R-CNN](https://arxiv.org/pdf/1504.08083v2.pdf) and [Faster R-CNN](https://arxiv.org/abs/1506.01497).

As most object detection methods, R-CNN approaches use a deep Neural Network which was trained for image classification using millions of annotated images and modify it for the purpose of object detection. The basic idea from the first R-CNN paper is illustrated in the Figure below (taken from the paper): (1) Given an input image, (2) in a first step, a large number region proposals are generated. (3) These region proposals, or Regions-of-Interests (ROIs), are then each independently sent through the network which outputs a vector of e.g. 4096 floating point values for each ROI. Finally, (4) a classifier is learned which takes the 4096 float ROI representation as input and outputs a label and confidence to each ROI.  
<p align="center">
<img src="media/rcnn_pipeline.jpg" width="600" align="center"/>
</p>

While this approach works well in terms of accuracy, it is very costly to compute since the Neural Network has to be evaluated for each ROI. Fast R-CNN addresses this drawback by only evaluating most of the network (to be specific: the convolution layers) a single time per image. According to the authors, this leads to a 213 times speed-up during testing and a 9x speed-up during training without loss of accuracy. Faster R-CNN then shows how ROIs can be computed as part of the network, essentially combining all steps in the figure above into a single DNN.


### Intersection-over-Union overlap metric

Often we want to measure by how much two given rectangles overlap. For example, one rectangle might correspond to the ground-truth location of an object, while the second rectangle is detected location, and the goal is to measure how precise (if at all) the object was detected.

For this, a metric called Intersection-over-Union (IoU) is typically used. In the example below, the IoU is given by dividing the yellow area by the yellow and blue area. See also this [page](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) for a more in-depth discussion.     
<p align="center">
<img src="media/iou_example.jpg" width="400" align="center"/>
</p>







### Non-maxima suppression
Object detection methods often output multiple detections which fully or partly cover the same object in an image. These detections need to be pruned to be able to count objects and obtain their exact locations in the image. This is traditionally done using a technique called Non-Maxima Suppression (NMS), and is implemented by iteratively selecting the detection with highest confidence and removing all other detections which (i) are classified to be of the same class; and (ii) significantly overlap measured using the Intersection-over-Union (IOU) metric.

Detection results with confidence scores before (left) and after non-maxima Suppression with (middle) conservative IOU threshold and (right) aggressive IOU threshold:
<p align="center">
<img src="media/nms_example.jpg" width="600" align="center"/>
</p>

### Mean Average Precision
Once trained, the quality of the model can be measured using different criteria, such as precision, recall, accuracy, area-under-curve, etc. A common metric which is used for the Pascal VOC object recognition challenge is to measure the Average Precision (AP) for each class. Average Precision takes confidence in the detections into account and hence assigns a smaller penalty to false detections with low confidence. For a description of Average Precision see [Everingham et. al](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf). The mean Average Precision (mAP) is computed by taking the average over all APs.
