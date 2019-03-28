(This document is up-to-date as of 3/27/2019)

# Overview of Azure's Computer Vision Offerings
[Microsoft Azure](https://azure.microsoft.com/en-us/) provides a variety of options when it comes to computer vision.
The outline below provides an overview of such services, starting with the highest level service where you simply
consume an API to the lowest level service where you develop the model and the infrastructure required to deploy it.

This document covers following topics:

* [What is Computer Vision](#What-is-Computer-Vision)
* [Computer Vision and Machine Learning Services in Azure](#Computer-Vision-and-Machine-Learning-Services-in-Azure)
    - [Cognitive Services](#Cognitive-Services)
    - [Custom Vision Service](#Custom-Vision-Service)
    - [Azure Machine Learning Service](#Azure-Machine-Learning-Service)
* [What Should I Use?](#What-Should-I-Use?)


## What is Computer Vision
Computer vision is one of the most popular disciplines in industry and academia nowadays that aims to train computers
to understand the visual content found in images and videos. With computer vision, computers can identify
visual content and label it with confidence, detect potentially harmful content, get location of thousands of objects
within an image, identify image types and color schemes in pictures, and do
[much more](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/home).

Click on the following topics to see more details:


<details>
<summary><strong>Image Classification</strong></summary>

A large number of problems in the computer vision domain can be solved using image classification approaches.
These include building models which answer questions such as, *"Is an OBJECT present in the image?"*
(where OBJECT could for example be "dog", "car", "ship", etc.) as well as more complex questions, like
*"What class of eye disease severity is evinced by this patient's retinal scan?"*

Image classification can be further categorized into **single-label** and **multi-label** classifications
depending on whether a target image contains a single object class or multiple objects of different classes.


<img src="https://cvbp.blob.core.windows.net/public/images/document_images/example_single_classification.png" width="600"/>

<i>An example of image classification modeling</i><br>

<img src="https://cvbp.blob.core.windows.net/public/images/document_images/example_multi_classification.png" width="600"/>

<i>An example of **multi-label** image classification model output</i><br>

</details>


<details>
<summary><strong>Image Similarity</strong></summary>

Retail companies want to show customers products which are similar to the ones bought in the past.
Or companies with large amounts of data want to organize and search their images effectively.
Image similarity detection can solve such interesting problems.

<img src="https://cvbp.blob.core.windows.net/public/images/document_images/example_image_similarity.jpg" width="600"/>

<i>An example of image similarity modeling</i><br>

</details>


<details>
<summary><strong>Object Detection</strong></summary>

Object Detection is one of the main problems in Computer Vision. Traditionally, this required expert knowledge to
identify and implement so called “features” that highlight the position of objects in the image.
Starting in 2012 with the famous [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf),
Deep Neural Networks are used to automatically find these features which lead to a huge improvement in the field
for a large range of problems.

<img src="https://cvbp.blob.core.windows.net/public/images/document_images/example_object_detection.png" width="600"/>

<i>An example of object detection model output</i><br>

</details>


<details>
<summary><strong>Image Segmentation</strong></summary>

In computer vision, the task of masking out pixels belonging to different classes of objects such as
water, barren and trees is referred to as image segmentation. Specifically, image segmentation is the process of
assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

<img src="https://cvbp.blob.core.windows.net/public/images/document_images/example_image_segmentation.png" width="600"/>

<i>An example of image segmentation</i><br>

</details>
