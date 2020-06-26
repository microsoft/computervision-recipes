# Multi-Object Tracking

```diff
+ June 2020: All notebooks/code in this directory is work-in-progress and might not fully execute. 
```

This directory provides examples and best practices for building multi-object tracking systems. Our goal is to enable the users to bring their own datasets and train a high-accuracytracking model easily. While there are many open-source trackers available, we have implemented the [FairMOT tracker](https://github.com/ifzhang/FairMOT) specifically, as its algorithm has shown competitive tracking performance in recent MOT benchmarking challenges, at fast inference speed.

## Technology
Multi-object-tracking (MOT) is one of the hot research topics in Computer Vision, due to its wide applications in autonomous driving, traffic surveillance, etc. It builds on object detection technology, in order to detect and track all objects in a dynamic scene over time. Inferring target trajectories correctly across successive image frames remains challenging: occlusion happens when objects overlap; the number of and appearance of objects can change. Compared to object detection algorithms, which aim to output rectangular bounding boxes around the objects, MOT algorithms additionally associated an ID number to each box to identify that specific object across the image frames.

As seen in the figure below ([Ciaparrone, 2019](https://arxiv.org/pdf/1907.12740.pdf)), a typical multi-object-tracking algorithm performs part or all of the following steps:
* Detection: Given the input raw image frames (step 1), the detector identifies object(s) on each image frame as bounding box(es) (step 2).
* Feature extraction/motion prediction: For every detected object, visual appearance and motion features are extracted (step 3). Sometimes, a motion predictor (e.g. Kalman Filter) is also added to predict the next position of each tracked target.
* Affinity: The feature and motion predictions are used to calculate similarity/distance scores between pairs of detections and/or tracklets, or the probabilities of detections belonging to a given target or tracklet (step 4).
* Association: Based on these scores/probabilities, a specific numerical ID is assigned to each detected object as it is tracked across successive image frames (step 5).

<p align="center">
<img src="./media/figure_motmodules2.jpg" width="700" align="center"/>
</p>


## State-of-the-art (SoTA)

### Tracking-by-detection (two-step) vs one-shot tracker  

Typical tracking algorithms address the detection and feature extraction processes in distinct successive steps. Recent research -[(Voigtlaender et al, 2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Voigtlaender_MOTS_Multi-Object_Tracking_and_Segmentation_CVPR_2019_paper.pdf), [(Wang et al, 2019)](https://arxiv.org/pdf/1909.12605.pdf), [(Zhang et al, 2020)](https://arxiv.org/pdf/1909.12605.pdf)- has moved onto combining the detection and feature embedding processes such that they are learned in a shared model (single network), particularly when both steps involving deep learning models. This framework is called single-shot or one-shot, and recent models include FairMOT  [(Zhang et al, 2020)](https://arxiv.org/pdf/1909.12605.pdf), JDE [(Wang et al, 2019)](https://arxiv.org/pdf/1909.12605.pdf) and TrackRCNN [(Voigtlaender et al, 2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Voigtlaender_MOTS_Multi-Object_Tracking_and_Segmentation_CVPR_2019_paper.pdf). Such single-shot models are more efficient than typical tracking-by-detection models and have shown faster inference speeds due to the shared computation of the single network representation of the detection and feature embedding: on the [MOT16 Challenge dataset](https://motchallenge.net/results/MOT16/), FAIRMOT and JDE achieve 30 fps and 18.5 fps respectively, while DeepSORT_2, a tracking-by-detection tracker with lower performance achieves 17.4 fps.

As seen in the table below, the FairMOT model  has a much improved tracking performance - MOTA, IDF1 (please see the [FAQ](FAQ.md) for more details on performance metrics)-, while the JDE model has a much worse ID-switch number [(Zhang et al, 2020)](https://arxiv.org/pdf/1909.12605.pdf). This is because the FairMOT model uses a typical anchor-based object detector network for feature embedding with a downsampled feature map, leading to a mis-alignment between the anchors and object center, hence re-iding issues. FairMOT solves these issues by: (i) estimating the object center instead of the anchors and using a higher resolution feature map for object detection and feature embedding, (ii) aggregating high-level and low-level features to handle scale variations across different sizes of objects.

<center>

| Tracker  | MOTA | IDF1 |	ID-Switch | fps |
| -------- | ---- | ---- | ---------  | --- |
|DeepSORT_2| 61.4 | 62.2 | 781 | 17.4 |
|JDE| 64.4 | 64.4 | 55.8 |1544 | 18.5 |
|FairMOT| 68.7 | 70.4 | 953 | 25.8 |

</center>


### Popular datasets

<center>

| Name  | Year  | Duration |	# tracks/ids | Scene | Object type |
| ----- | ----- | -------- | --------------  | ----- |  ---------- |
| [MOT15](https://arxiv.org/pdf/1504.01942.pdf)| 2015 | 16 min | 1221 | Outdoor | Pedestrians |
| [MOT16/17](https://arxiv.org/pdf/1603.00831.pdf)| 2016 | 9 min | 1276 | Outdoor & indoor | Pedestrians & vehicles |
| [CVPR19/MOT20](https://arxiv.org/pdf/1906.04567.pdf)| 2019 | 26 min | 3833 | Crowded scenes | Pedestrians & vehicles |
| [PathTrack](http://openaccess.thecvf.com/content_ICCV_2017/papers/Manen_PathTrack_Fast_Trajectory_ICCV_2017_paper.pdf)| 2017 | 172 min | 16287 | Youtube people scenes | Persons |
| [Visdrone](https://arxiv.org/pdf/1804.07437.pdf)| 2019 | - | - | Outdoor view from drone camera | Pedestrians & vehicles |
| [KITTI](http://www.jimmyren.com/papers/rrc_kitti.pdf)| 2012 | 32 min | - | Traffic scenes from car camera | Pedestrians & vehicles |
| [UA-DETRAC](https://arxiv.org/pdf/1511.04136.pdf) | 2015 | 10h | 8200 | Traffic scenes | Vehicles |
| [CamNeT](https://vcg.ece.ucr.edu/sites/g/files/rcwecm2661/files/2019-02/egpaper_final.pdf) | 2015 | 30 min | 30 | Outdoor & indoor | Persons |

</center>

### Popular publications

| Name | Year | MOT16 IDF1 | MOT16 MOTA | Inference Speed(fps) | Online/ Batch | Detector |  Feature extraction/ motion model | Affinity & Association Approach |
| ---- | ---- | ---------- | ---------- | -------------------- | ------------- | -------- | -------------------------------- | -------------------- |
|[A Simple Baseline for Multi-object Tracking -FairMOT](https://arxiv.org/pdf/2004.01888.pdf)|2020|70.4|68.7|25.8|Online|One-shot tracker with detector head|One-shot tracker with re-id head & multi-layer feature aggregation, IOU, Kalman Filter| JV algorithm on IOU, embedding distance,
|[How to Train Your Deep Multi-Object Tracker -DeepMOT-Tracktor](https://arxiv.org/pdf/1906.06618v2.pdf)|2020|53.4|54.8|1.6|Online|Single object tracker: Faster-RCNN (Tracktor), GO-TURN, SiamRPN|Tracktor, CNN re-id module|Deep Hungarian Net using Bi-RNN|
|[Tracking without bells and whistles -Tracktor](https://arxiv.org/pdf/1903.05625.pdf)|2019|54.9|56.2|1.6|Online|Modified Faster-RCNN| Temporal bbox regression with bbox camera motion compensation, re-id embedding from Siamese CNN| Greedy heuristic to merge tracklets using re-id embedding distance|
|[Towards Real-Time Multi-Object Tracking -JDE](https://arxiv.org/pdf/1909.12605v1.pdf)|2019|55.8|64.4|18.5|Online|One-shot tracker - Faster R-CNN with FPN|One-shot - Faster R-CNN with FPN, Kalman Filter|Hungarian Algorithm|
|[Exploit the connectivity: Multi-object tracking with TrackletNet -TNT](https://arxiv.org/pdf/1811.07258.pdf)|2019|56.1|49.2|0.7|Batch|MOT challenge detections|CNN with bbox camera motion compensation, embedding feature similarity|CNN-based similarity measures between tracklet pairs; tracklet-based graph-cut optimization|
|[Extending IOU based Multi-Object Tracking by Visual Information -VIOU](http://elvera.nue.tu-berlin.de/typo3/files/1547Bochinski2018.pdf)|2018|56.1(VisDrone)|40.2(VisDrone)|20(VisDrone)|Batch|Mask R-CNN, CompACT|IOU|KCF to merge tracklets using greedy IOU heuristics|
|[Simple Online and Realtime Tracking with a Deep Association Metric -DeepSORT](https://arxiv.org/pdf/1703.07402v1.pdf)|2017|62.2| 61.4|17.4|Online|Modified Faster R-CNN|CNN re-id module, IOU, Kalman Filter|Hungarian Algorithm, cascaded approach using Mahalanobis distance (motion), embedding distance |
|[Multiple people tracking by lifted multicut and person re-identification -LMP](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tang_Multiple_People_Tracking_CVPR_2017_paper.pdf)|2017|51.3|48.8|0.5|Batch|[Public detections](https://arxiv.org/pdf/1610.06136.pdf)|StackeNetPose CNN re-id module|Spatio-temporal relations, deep-matching, re-id confidence; detection-based graph lifted-multicut optimization|


## Notebooks

We provide several notebooks to show how multi-object-tracking algorithms can be designed and evaluated:

| Notebook name | Description |
| --- | --- |
| [00_webcam.ipynb](./00_webcam.ipynb)| Quick-start notebook which demonstrates how to build an object tracking system using a single video or webcam as input.
| [01_training_introduction.ipynb](./01_training_introduction.ipynb)| Notebook which explains the basic concepts around model training, inferencing, and evaluation using typical tracking performance metrics.|
| [02_mot_challenge.ipynb](./02_mot_challenge.ipynb) | Notebook which runs inference on a large dataset, the MOT challenge dataset. |



## Frequently asked questions

Answers to frequently asked questions such as "How does the technology work?", "What data formats are required?" can be found in the [FAQ](FAQ.md) located in this folder. For generic questions such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see the [FAQ.md](../classification/FAQ.md) in the classification folder.



## Contribution guidelines

See the [contribution guidelines](../../CONTRIBUTING.md) in the root folder.
