# Action Recognition

This is a place holder. Content will follow soon.

![](./media/action_recognition.gif)

*Example of action recognition*

## Overview

| Folders |  Description |
| -------- |  ----------- |
| [i3d](i3d)  | Scripts for fine-tuning a pre-trained Two-Stream Inflated 3D ConvNet (I3D) model on the HMDB-51 dataset
| [video_annotation](video_annotation)  | Instructions and helper functions to annotate the start and end position of actions in video footage|

## Functionality

In [i3d](i3d) we show how to fine-tune a Two-Stream Inflated 3D ConvNet (I3D) model. This model was introduced in \[[1](https://arxiv.org/pdf/1705.07750.pdf)\] and achieved state-of-the-art in action classification on the HMDB-51 and UCF-101 datasets. The paper demonstrated the effectiveness of pre-training action recognition models on large datasets - in this case the Kinetics Human Action Video dataset consisting of 306k examples and 400 classes. We provide code for replicating the results of this paper on HMDB-51. We use models pre-trained on Kinetics from [https://github.com/piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d). Evaluating the model on the test set of the HMDB-51 dataset (split 1) using [i3d/test.py](i3d/test.py) should yield the following results:

| Model | Paper top 1 accuracy (average over 3 splits) | Our models top 1 accuracy (split 1 only) |
| ------- | -------| ------- |
| RGB | 74.8 | 73.7 |
| Optical flow | 77.1 | 77.5 |
| Two-Stream | 80.7 | 81.2 |

In order to train an action recognition model for a specific task, annotated training data from the relevant domain is needed. In [video_annotation](video_annotation), we provide tips and examples for how to use a best-in-class video annotation tool ([VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/)) to label the start and end positions of actions in videos.

## State-of-the-art

In the tables below, we list datasets which are commonly used and also give an overview of the state-of-the-art. Note that the information below is reasonably exhaustive and should cover most major publications until 2018. Expect however some level of incompleteness and slight incorrectness (e.g. publication year being off by plus/minus 1 year due) since the tables below were mainly compiled to give a high-level picture of where the field is and how it evolved over the last years.

Recommended reading:
- As introduction to action recognition the blog [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review).
- [ActionRecognition.net](http://actionrecognition.net/files/dset.php) for the latest state-of-the-art accuracies on popular research benchmark datasets.
- All papers highlighted in yellow in the publications table below.

Popular datasets:

| Name  | Year  |  Number of classes |	#Clips |	Average length per video | Notes |
| ----- | ----- | ----------------- | ------- | -------------------------  |  ----------- |
| KTH   | 2004| 6| 600| |  | |
|Weizmann|	2005|	9|	81|	|	 |
|HMDB-51| 2011|	51|	6.8k| |	 |
|UCF-101|	2012|	101|	13.3k|	7 sec (min: 1sec, max: 71sec)|	|
|Sports-1M|	2014|	487|	1M| | |
|THUMOS14|	2014|	101|	18k|	(total: 254h)|	Dataset for temporal action |
|ActivityNet|	2015|	200|	28.1k|	? 1 min 40 sec|	|
|Charades|	2016|	157|	66.5k from 9848 videos|	Each video (not action) is 30 seconds	| Daily tasks, classification and temporal localization challenges|
|Youtube-8M|	2016|	4800 | |  | Not an action dataset, but rather a classification one (ie what objects occur in each video). Additional videos added in 2018.|
|Kinetics-400|	2017|	400|	306k|	10 sec|	|
|Kinetics-600|	2018|	600|	496k|   |
|Something-Something|	2017|	174|	110k|	2-6 sec	| Low level actions, e.g. "pushing something left to right". Additional videos added in 2019.|
|AVA|	2018|	80|	1.6M in 430 videos|	Each video is 15min long with 1 frame annotated per second with location of person, and for each  person one of 80 "atomic" actions. Combine people annotations into tracks.
|Youtube-8M Segments|	2019|	1000|	237k|	5sec|	Used for localization Kaggle challenge. Think focuses on objects, not actions.|



Popular publications, with recommended papers to read highlighted in yellow:
<img align="center" src="./media/publications.png"/>  


Most pulications focus on accuracy rather than on inferencing speed. The paper "Representation Flow for Action Recognition" is a noteworthy exception with this figure:
<img align="center" src="./media/inference_speeds.png" width = "500"/>  

\[1\] J. Carreira and A. Zisserman. Quo vadis, action recognition?
a new model and the kinetics dataset. In CVPR, 2017.