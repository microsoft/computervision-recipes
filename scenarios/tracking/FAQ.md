# Multi-Object Tracking

```diff
+ June 2020: This work is ongoing.
```

## Frequently asked questions

This document tries to answer frequent questions related to multi-object tracking. For generic Machine Learning questions, such as "How many training examples do I need?" or "How to monitor GPU usage during training?" see also the image classification [FAQ](https://github.com/microsoft/ComputerVision/blob/master/classification/FAQ.md).

* General
  * [Why FairMOT repository for the tracking algorithm?](#why-FAIRMOT)
  * [What are additional complexities that can enhance the current MOT algorithm](#What-are-additional-complexities-that-can-enhance-the-current-MOT-algorithm)
  * [What is the difference between online and offline (batch) tracking algorithms?](#What-is-the-difference-between-online-and-offline-tracking-algorithms)

* Data  
  * [How to annotate a video for evaluation?](#how-to-annotate-a-video-for-evaluation)
  * [What is the MOT Challenge format used by the evaluation package?](#What-is-the-MOT-Challenge-format-used-by-the-evaluation-package)

* Technology State-of-the-Art (SoTA)
  * [What is the architecture of the FairMOT tracking algorithm?](#What-is-the-architecture-of-the-FairMOT-tracking-algorithm)
  * [What are SoTA object detectors used in tracking-by-detection trackers?](#What-are-SoTA-object-detectors-used-in-tracking-by-detection-trackers) 
  * [What are SoTA feature extraction techniques used in tracking-by-detection trackers?](#What-are-SoTA-feature-extraction-techniques-used-in-tracking-by-detection-trackers)
  * [What are SoTA affinity and association techniques used in tracking-by-detection trackers?](#What-are-SoTA-affinity-and-association-techniques-used-in-tracking-by-detection-trackers)
  * [What are the main evaluation metrics for tracking performance?](#What-are-the-main-evaluation-metrics)

* Training and Inference
  * [How to improve training accuracy?](#how-to-improve-training-accuracy)
  * [What are the main training parameters in FairMOT](#what-are-the-main-training-parameters-in-FairMOT)
  * [What are the main inference parameters in FairMOT?](#What-are-the-main-inference-parameters-in-FairMOT])
  * [What are the training losses for MOT using FairMOT?](#What-are-the-training-losses-for-MOT-using-FairMOT? )

* MOT Challenge
  * [What is the MOT Challenge?](#What-is-the-MOT-Challenge)




## General

### Why FairMOT?
FairMOT is an [open-source](https://github.com/ifzhang/FairMOT) online one-shot tracking algorithm, that has shown [competitive performance in recent MOT benchmarking challenges](https://motchallenge.net/method/MOT=3015&chl=5), at fast inference speed.  


### What are additional complexities that can enhance the current MOT algorithm?
Multi-camera processing, and compensation for camera-movement effect on association features with epipolar geometry.  


### What is the difference between online and offline tracking algorithms? 
These algorithms differ at the data association step. In online tracking, the detections in a new frame are associated with tracks generated previously from previous frames, thus existing tracks are extended or new tracks are created. In offline (batch) tracking , all observations in a batch of frames are considered globally (see figure below), i.e. they are linked together into tracks by obtaining a global optimal solution. Offline tracking can perform better with tracking issues such as long-term occlusion, or similar targets that are spatially close. However, offline tracking is slow, hence not suitable for online tasks such as for autonomous driving. Recently, research has focused on online tracking algorithms, which have reached the performance of offline-tracking while still maintaining high inference speed. 

<p align="center">
<img src="./media/fig_onlineBatch.jpg" width="400" align="center"/>
</p>

## Data

### How to annotate a video for evaluation?
We can use an annotation tool, such as VOTT, to annotate a video for ground-truth. For example, for the evaluation video, we can draw bounding boxes around the 2 cans, and tag them as `can_1` and `can_2`: 
<p align="center">
<img src="./media/carcans_vott_ui.jpg" width="600" align="center"/>
</p>

Before annotating, make sure to set the extraction rate to match that of the video. After annotation, you can export the annotation results into csv form. You will end up with the extracted frames as well as a csv file containing the bounding box and id info: ``` [image] [xmin] [y_min] [x_max] [y_max] [label]```

### What is the MOT Challenge format used by the evaluation package?
The evaluation package, from  the [py-motmetrics](https://github.com/cheind/py-motmetrics) repository, requires the ground-truth data to be in [MOT challenge](https://motchallenge.net/) format, i.e.: 
```
[frame number] [id number] [bbox left] [bbox top] [bbox width] [bbox height][confidence score][class][visibility]
```
The last 3 columns can be set to -1 by default, for the purpose of ground-truth annotation.


## Technology State-of-the-Art (SoTA)


### What is the architecture of the FairMOT tracking algorithm?
It consists of a single encoder-decoder neural network which extracts high resolution feature maps of the image frame. As a one-shot tracker, these feed into two parallel heads for predicting bounding boxes and re-id features respectively, see [source](https://arxiv.org/pdf/2004.01888v2.pdf): 
<p align="center">
<img src="./media/figure_fairMOTarc.jpg" width="800" align="center"/>
</p>

<center>

Source: [Zhang, 2020](https://arxiv.org/pdf/2004.01888v2.pdf)

</center>


### What are SoTA object detectors used in tracking-by-detection trackers?
The most popular object detectors used by SoTA tacking algorithms include: [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf), [SSD](https://arxiv.org/pdf/1512.02325.pdf) and [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf). Please see our [object detection FAQ page](../detection/faq.md) for more details.  


### What are SoTA feature extraction techniques used in tracking-by-detection trackers?
While older algorithms used local features such as optical flow or regional features (e.g. color histograms, gradient-based features or covariance matrix), newer algorithms have a deep-learning based feature representation. The most common deep-learning approaches use classical CNN to extract visual features, typically trained on re-id datasets, such as the [MARS dataset](http://www.liangzheng.com.cn/Project/project_mars.html). The following figure is an example of a CNN used for MOT by the [DeepSORT tracker](https://arxiv.org/pdf/1703.07402.pdf):
        <p align="center">
        <img src="./media/figure_DeepSortCNN.jpg" width="600" align="center"/>
        </p>
Newer deep-learning approaches include Siamese CNN networks, LSTM networks, or CNN with correlation filters. In Siamese CNN networks, a pair of CNN networks is used to measure similarity between two objects, and the CNNs are trained with loss functions that learn features that best differentiates them. 
             <p align="center">
            <img src="./media/figure_SiameseNetwork.jpg" width="400" align="center"/>
            </p>
<center>

 Source: [(Simon-Serra et al, 2015)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Simo-Serra_Discriminative_Learning_of_ICCV_2015_paper.pdf)

</center>

In LSTM network, extracted features from different detections in different time frames are used as inputs to a LSTM network, which predicts the bounding box for the next frame based on the input history.
             <p align="center">
            <img src="./media/figure_LSTM.jpg" width="550" align="center"/>
            </p>
<center>

Source: [Ciaparrone, 2019](https://arxiv.org/pdf/1907.12740.pdf)

</center>

Correlation filters can also be convolved with feature maps from CNN network to generate a prediction of the target's location in the next time frame. This was done by [Ma et al](https://faculty.ucmerced.edu/mhyang/papers/iccv15_tracking.pdf) as follows:
            <p align="center">
            <img src="./media/figure_CNNcorrFilters.jpg" width="500" align="center"/>
            </p>


### What are SoTA affinity and association techniques used in tracking-by-detection trackers? 
Simple approaches use similarity/affinity scores calculated from distance measures over features extracted by the CNN to optimally match object detections/tracklets with established object tracks across successive frames. To do this matching,  Hungarian (Huhn-Munkres) algorithm is often used for online data association, while K-partite graph global optimization techniques are used for offline data association. 

In more complex deep-learning approaches, the affinity computation is often merged with feature extraction. For instance, [Siamese CNNs](https://arxiv.org/pdf/1907.12740.pdf) and [Siamese LSTMs](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w21/Wan_An_Online_and_CVPR_2018_paper.pdf) directly output the affinity score.


### What are the main evaluation metrics?
As multi-object-tracking is a complex CV task, there exists many different metrics to evaluate the tracking performance.  Based on how they are computed, metrics can be event-based [CLEARMOT metrics](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf) or [id-based metrics](https://arxiv.org/pdf/1609.01775.pdf). The main metrics used to gauge performance in the [MOT benchmarking challenge](https://motchallenge.net/results/MOT16/) include MOTA, IDF1, and ID-switch.
* MOTA (Multiple Object Tracking Accuracy): it gauges overall accuracy performance, with event-based computation of how often mismatch occurs between the tracking results and ground-truth. MOTA contains the counts of FP (false-positive), FN(false negative) and id-switches (IDSW), normalized over the total number of ground-truth (GT) tracks.
<p align="center">
<img src="./media/eqn_mota.jpg" width="200" align="center"/>
</p>

* IDF1: gauges overall performance, with id-based computation of how long the tracker correctly identifies the target. It is the harmonic mean of identification precision (IDP) and recall (IDR): 
<p align="center">
<img src="./media/eqn_idf1.jpg" width="450" align="center"/>
</p>

* ID-switch: when the tracker incorrectly changes the ID of the trajectory. This is illustrated in the following figure: in the left box, person A and person B overlap and are not detected and tracked in frames 4-5. This results in an id-switch in frame 6, where person A is attributed the ID_2, which previously tagged person B. In another example in the right box, the tracker loses track of person A (initially identified as ID_1) after frame 3, and eventually identifies that person with a new ID (ID_2) in frame n, showing another instance of id-switch. 

<p align="center">
<img src="./media/fig_tracksEval.jpg" width="600" align="center"/>
</p>




## Training and inference


### What are the main training parameters in FairMOT?
The main training parameters include batch size, learning rate and number of epochs. Additionally, FairMOT uses Torch's Adam algorithm as the default optimizer.


### How to improve training accuracy?
One can improve the training procedure by modifying the learning rate and number of epochs.

### What are the training losses for MOT using FairMOT?
Losses generated by the FairMOT include detection-specific losses (e.g. hm_loss, wh_loss, off_loss) and id-specific losses (id_loss). The overall loss (loss) is a weighted average of the detection-specific and id-specific losses, see the [FairMOT paper](https://arxiv.org/pdf/2004.01888v2.pdf). 

### What are the main inference parameters in FairMOT?
- input_w and input_h: image resolution of the dataset video frames;
- conf_thres, nms_thres, min_box_area: these thresholds used to filter out detections that do not meet the confidence level, nms level and size as per the user requirement;
- track_buffer: if a lost track is not matched for some number of frames as determined by this threshold, it is deleted, i.e. the id is not reused.

## MOT Challenge

### What is the MOT Challenge?
It hosts the most common benchmarking datasets for pedestrian MOT. Different datasets exist: MOT15, MOT16/17, MOT 19/20. These datasets contain many video sequences, with different tracking difficulty levels, with annotated ground-truth. Detections are also provided for optional use by the participating tracking algorithms.
