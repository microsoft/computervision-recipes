# Multi-Object Tracking

```diff
+ July 2020: Functionality in this directory is work-in-progress; some notebooks may be incomplete.
```

This directory provides examples and best practices for building and inferencing multi-object tracking systems. Our goal is to enable users to bring their own datasets and to train a high-accuracy tracking model with ease. While there are many open-source trackers available, we have integrated the [FairMOT tracker](https://github.com/ifzhang/FairMOT) to this repository. The FairMOT algorithm has shown competitive tracking performance in recent MOT benchmarking challenges, while also having respectable inference speeds.


## Notebooks

We provide several notebooks to show how multi-object-tracking algorithms can be designed and evaluated:

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./00_webcam.ipynb)| Quick-start notebook that demonstrates how to build an object tracking system using a single video or webcam as input.
| [01_training_introduction.ipynb](./01_training_introduction.ipynb)| Notebook that explains the basic concepts around model training, inferencing, and evaluation using typical tracking performance metrics.|
| [02_mot_challenge.ipynb](./02_mot_challenge.ipynb) | Notebook that runs model inference on the commonly used MOT Challenge dataset. |


## Technology
Due to its applications in autonomous driving, traffic surveillance, etc., multi-object-tracking (MOT) is a popular and growing area of reseach within Computer Vision. MOT builds on object detection technology to detect and track objects in a dynamic scene over time. Inferring target trajectories correctly across successive image frames remains challenging. For example, occlusion can cause the number and appearance of objects to change, resulting in complications for MOT algorithms. Compared to object detection algorithms, which aim to output rectangular bounding boxes around the objects, MOT algorithms additionally associated an ID number to each box to identify that specific object across the image frames.

As seen in the figure below ([Ciaparrone, 2019](https://arxiv.org/pdf/1907.12740.pdf)), a typical multi-object-tracking algorithm performs part or all of the following steps:
* Detection: Given the input raw image frames (step 1), the detector identifies object(s) in each image frame as bounding box(es) (step 2).
* Feature extraction/motion prediction: For every detected object, visual appearance and motion features are extracted (step 3). A motion predictor (e.g. Kalman Filter) is occasionally also added to predict the next position of each tracked target.
* Affinity: The feature and motion predictions are used to calculate similarity/distance scores between pairs of detections and/or tracklets, or the probabilities of detections belonging to a given target or tracklet (step 4).
* Association: Based on these scores/probabilities, a specific numerical ID is assigned to each detected object as it is tracked across successive image frames (step 5).

<p align="center">
<img src="./media/figure_motmodules2.jpg" width="700" align="center"/>
</p>

## Frequently Asked Questions

Answers to frequently asked questions, such as "How does the technology work?" or "What data formats are required?", can be found in the [FAQ](FAQ.md) located in this folder. For generic questions, such as "How many training examples do I need?" or "How to monitor GPU usage during training?", see the [FAQ.md](../classification/FAQ.md) in the classification folder.

## Contribution Guidelines

See the [contribution guidelines](../../CONTRIBUTING.md) in the root folder.
