# Action Recognition

This directory contains resources for building video-based action recognition systems. Our goal is to enable users to easily and quickly train highly accurate and fast models on their own custom datasets.

Action recognition (also known as activity recognition) consists of classifying various actions from a sequence of frames, such as "reading" or "drinking":

![](./media/action_recognition2.gif "Example of action recognition")


## Notebooks

The following example notebooks are provided:

| Notebook | Description |
| --- | --- |
| [00_webcam](00_webcam.ipynb) | Real-time inference example on Webcam input. |
| [01_training_introduction](01_training_introduction.ipynb) | Introduction to action recognition: training, evaluating, predicting |
| [01_training_introduction](02_training_hmdb.ipynb) | Fine-tuning on the HMDB-51 dataset. |
| [02_video_transformation](10_video_transformation.ipynb) | Examples of video transformations. |

Furthermore, tools for data annotation are located in the [video_annotation](./video_annotation) subfolder.


## Technology

Action recognition is an active field of research, with large number of approaches being published every year. One of the approaches which stands out is the  **R(2+1)D** model which is described in the 2019 paper "[Large-scale weakly-supervised pre-training for video action recognition](https://arxiv.org/abs/1905.00561)".

R(2+1)D is highly accurate and at the same time significantly faster than other approaches:
- Its accuracy comes in large parts from an extra pre-training step which uses 65 million automatically annotated video clips.
- Its speed comes from simply using video frames as input. Many other state-of-the-art methods require optical flow fields to be pre-computed which is computationally expensive (see the "Inference speed" section below).

We base our implementation and pretrained weights on this [github](https://github.com/moabitcoin/ig65m-pytorch) repository, with added functionality to make training and evaluating custom models more user-friendly. We use the IG-Kinetics dataset for pre-training, however the currently only published results on the HMDB-51 dataset use the much smaller (and less noisy) Kinetics dataset. Nevertheless, the results below show that our implementation is able to achieve and push state-of-the-art accuracy on HMDB-51: 

| Model | Pre-training dataset | Reported in the paper | Our results |
| ------- | -------| ------- | ------- |
| R(2+1)D | Kinetics | 74.5% |  |
| R(2+1)D | IG-Kinetics |  | 79.8% |


## State-of-the-art

Popular benchmark datasets in the field, as well as state-of-the-art publications are listed below. Note that the information is reasonably exhaustive and should cover many of the major publications until 2018. Expect however some level of incompleteness and slight incorrectness (e.g. publication year being off by plus/minus a year).

We recommend the following reading to familiarize oneself with the field:
- As introduction to action recognition the blog [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review).
- [ActionRecognition.net](http://actionrecognition.net/files/dset.php) for the latest state-of-the-art accuracies on popular research benchmark datasets.
- The three papers with links in the publications table below.

#### Popular datasets

| Name  | Year  |  Number of classes |	#Clips |
| ----- | ----- | ----------------- | ------- |
| KTH   | 2004| 6| 600|   
|Weizmann|	2005|	9|	81|		
|HMDB-51| 2011|	51|	6.8k| 	 
|UCF-101|	2012|	101|	13.3k|
|Sports-1M|	2014|	487|	1M|
|ActivityNet|	2015|	200|	28.1k|
|Charades|	2016|	157|	66.5k from 9848 videos|
|Kinetics-400|	2017|	400|	306k|
|Something-Something|	2017|	174|	110k|
|Kinetics-600|	2018|	600|	496k|  
|AVA|	2018|	80|	1.6M from 430 videos|
|Youtube-8M Segments|	2019|	1000|	237k|
|IG-Kinetics|	2019|	359|	65M|


#### Popular publications

|                                                                                       | Year | UCF101 accuracy | HMDB51 accuracy | Kinetics accuracy | Pre-training on                                                              |
|---------------------------------------------------------------------------------------|------|-----------------|-----------------|-------------------|------------------------------------------------------------------------------|
| Learning Realistic Human Actions from Movies                                          | 2008 |                 |                 |                   | -                                                                            |
| Action Recognition with Improved Trajectories                                         | 2013 |                 | 57%             |                   | -                                                                            |
| 3D Convolutional Neural Networks for Action Recognition                               | 2013 |                 |                 |                   | -                                                                            |
| Two-Stream Convolutional Networks for Action Recognition in Videos                    | 2014 | 86%             | 58%             |                   | Combined UCF101 and HMDB51                                                   |
| Large-scale Video Classification with CNNs                                            | 2014 | 65%             |                 |                   | Sports-1M                                                                    |
| Beyond Short Snippets: Deep Networks for Video Classification                         | 2015 | 88%             |                 |                   | Sports-1M                                                                    |
| Learning Spatiotemporal Features with 3D Convolutional Networks                       | 2015 | 85%             |                 |                   | Sports-1M                                                                    |
| Initialization Strategies of Spatio-Temporal CNNs                                     | 2015 | 78%             |                 |                   | ImageNet                                                                     |
| Temporal Segment Networks: Towards Good Practices for Deep Action Recognition         | 2016 | 94%             | 69%             |                   | ImageNet                                                                     |
| Convolutional two-stream Network Fusion for Video Action Recognition                  | 2016 | 91%             | 58%             |                   | -                                                                            |
| [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)  **I3D model**               | 2017 | 98%             | 81%             | 74%               | Kinetics (+ImageNet)                                                         |
| Hidden Two-Stream Convolutional Networks for Action Recognition                       | 2017 | 97%             | 79%             |                   |                                                                              |
| Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification | 2017 | 93%             | 64%             | 62%               | Kinetics (+ImageNet)                                                         |
| End-to-End Learning of Motion Representation for Video Understanding (TVNet)          | 2018 | 95%             | 71%             |                   | ImageNet                                                                     |
| ActionFlowNet: Learning Motion Representation for Action Recognition                  | 2018 | 84%             | 56%             |                   | Optical-flow dataset                                                         |
| [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)  **R(2+1)D model**                | 2018 | 97%             | 79%             | 74%               | Kinetics                                                                     |
| Rethinking Spatiotemporal Feature Learning For Video Understanding,                   | 2018 | 97%             | 76%             | 77%               |                                                                              |
| Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?               | 2018 |                 |                 |                   |                                                                              |
| [Large-scale weakly-supervised pre-training for video action recognition](https://arxiv.org/abs/1905.00561)  **R(2+1)D model**          | 2019 |                 |                 | 81%               | 65 million automatically labeled web-videos (not publicly available) |
| Representation Flow for Action Recognition                                            | 2019 |                 | 81%             | 78%               | Kinetics                                                                     |
| Dance with Flow: Two-in-One Stream Action Recognition                                 | 2019 | 92%             |                 |                   | ImageNet                                                                     |


#### Inference speed

Most publications focus on accuracy rather than inference speed. The figure below from the paper "[Representation Flow for Action Recognition](https://arxiv.org/abs/1810.01455)" is a noteworthy exception. Note how fast R(2+1)D is with 471ms, compared especially to approaches which require optical flow fields as input to the DNN ("Flow" or "Two-stream").

<img align="center" src="./media/inference_speeds.png" width = "500" />  

## Coding guidelines

See the [coding guidelines](../../CONTRIBUTING.md#coding-guidelines) in the root
