# Action Recognition

This is a place holder. Content will follow soon.

## Overview

Also give as example either a screenshot of Jun Ki's demo, or play here his recording

| Folders |  Description |
| -------- |  ----------- |
| [video_annotation](video_annotation)  | Instructions and helper functions to annotate the start and end position of actions in video footage|

## Functionality and value-add

What our code can do, and what was missing from the repos we use.

## State-of-the-art

In the tables below, we list datasets which are commonly used and also give an overview of the state-of-the-art. Note that the tables are exhaustive and should cover most major publications until 2018. Expect however some level of incompleteness and slight incorrectness (e.g. publication year being off by plus/minus 1 year due, etc)

  also caused by "rounding/summarizing" the information rather than providing a detailed description.)

Recommended reading:
- As introduction to action recognition the blog [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review).
- [ActionRecognition.net](http://actionrecognition.net/files/dset.php) for the latest state-of-the-art accuracies on popular research benchmark datasets.
- All papers highlighted in **bold** in the publications table below.

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



Popular publications, with recommended papers to read highlighed in **bold**:



## Etc
