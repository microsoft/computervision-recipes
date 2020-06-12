# Multi-Object Tracking

This directory provides examples and best practices for building multi-object tracking systems. Our goal is to enable the users to bring their own datasets and train a high-accuracy model easily. 

## Technology
Multi-object-tracking (MOT) is one of the most active fields in Computer Vision. It builds on object detection technology, in order to detect and track all objects in a dynamic scene over time.  

 A typical multi-object-tracking algorithm performs the following:
* Detection: identify objects as bounding boxes
* Feature extraction/motion prediction: extract appearance, motion and/or interaction features
* Affinity: use feature and motion predictions to compute similarity score between pairs of detections
* Association: similarity measures are used to associate same target

As seen in the figure below (Ciaparrone, 2019), ...

<p align="center">
<img src="./media/figure_MOTmodules.png" width="600" align="center"/>
</p> 

TOADD on above

**Online vs offline(batch tracking)**
In general, MOT algorithms can be divided into onlne and offline tracking. In online tracking, only the observations up to the current frame are taken into consideration when processing a new frame: typically, the new detections in the new frame are associated with tracks generated previously from the previous frame, thus existing tracks are extended or new tracks are created. In offline tracking, all observations in a batch of frames are considered globally, in that they are linked together into tracks by obtaining a global optimal solution. In this way, offline tracking can perform better with tracking issues such as long-term occlusion, or similar targets that are spatially close. However, offline tracking is slow not suitable for online tasks, such as for autonomous driving. In recent, research has focused on online tracking algorithms, which have reached the performance of online tracking, while still maintaining high inference speed. 

<p align="center">
<img src="./media/fig_onlineBatch.png" width="400" align="center"/>
</p>

**Tracking-by-detection (two-step) vs one-shot tracker**

TOADD
FIND/DO diagrams

## State-of-the-art

### MOT modules 
* Detection:
    * Faster R-CNN, YOLO, SSD..
* Feature extraction & motion prediction:
    * Siamse CNN networks
    * LSTM netowrks
    * CNN + Correlation filters
* Affinity & association:
    * Similarity/affinity scores
    * Data association: matching object detections/tracklets with established object tracks from frame `t` to `t+1`. ToADD: hungarian algorithm...
    * Data association: classical (probabilistic) vs graph-based approaches. TOADD

### Evaluation metrics
As multi-object-tracking is a complex CV task, there exists many different metrics to evaluate the tracking performance. The main ones are:
* MOTA
* IDF1
* ID-switch

<p align="center">
<img src="./media/fig_tracksEval.png" width="400" align="center"/>
</p>


### Popular datasets
<p align="center">
<img src="./media/table_datasets.png" width="900" align="center"/>
</p>

### Popular publications
* General overview
* Benchmarking
* Seminal/baseline method papers

TOADD: publication lists + deep-dive comments

<p align="center">
<img src="./media/table_onlinePub.png" width="900" align="center"/>
</p>

<p align="center">
<img src="./media/table_offlinePub.png" width="900" align="center"/>
</p>

## Notebooks

We provide several notebooks to show how multi-object-tracking algorithms can be designed and evaluated:

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./00_webcam.ipynb)| Quick-start notebook which demonstrates how to build an object tracking system using a single video or webcam as input.
| [01_training_introduction.ipynb](./01_training_introduction.ipynb)| Notebook which explains the basic concepts around model training, inferencing, and evaluation using typical tracking performance metrics.|
| [02_mot_challenge.ipynb](./02_mot_challenge.ipynb) | Notebook which runs inference on a large dataset, the MOT challenge XX dataset. |



## Frequently asked questions

Answers to frequently asked questions such as "How does the technology work?", "What data formats are required?" can be found in the [FAQ](FAQ.md) located in this folder. For generic questions such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see the [FAQ.md](../classification/FAQ.md) in the classification folder.



## Contribution guidelines

See the [contribution guidelines](../../CONTRIBUTING.md) in the root folder.
