# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import itertools
import json
from typing import Callable, List, Tuple, Union, Generator, Optional, Dict

from pathlib import Path
import shutil

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from .references.engine import train_one_epoch, evaluate
from .references.coco_eval import CocoEvaluator
from .references.pycocotools_cocoeval import compute_ap
from .bbox import bboxes_iou, DetectionBbox
from ..common.gpu import torch_device


def _extract_od_results(
    pred: Dict[str, np.ndarray],
    labels: List[str],
    im_path: Union[str, Path] = None,
) -> Dict:
    """ Gets the bounding boxes, masks and keypoints from the prediction object.

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
            or MaskRCNN model, detached in the form of numpy array
        labels: list of labels without "__background__".
        im_path: the image path of the preds

    Return:
        a dict of DetectionBboxes, masks and keypoints
    """
    pred_labels = pred["labels"].tolist()
    pred_boxes = pred["boxes"].tolist()
    pred_scores = pred["scores"].tolist()

    det_bboxes = []
    for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
        label_name = labels[label - 1]
        det_bbox = DetectionBbox.from_array(
            box,
            score=score,
            label_idx=label,
            label_name=label_name,
            im_path=im_path,
        )
        det_bboxes.append(det_bbox)

    out = {"det_bboxes": det_bboxes, "im_path": im_path}

    if "masks" in pred:
        out["masks"] = pred["masks"].squeeze(1)

    if "keypoints" in pred:
        out["keypoints"] = pred["keypoints"]

    return out


def _apply_threshold(
    pred: Dict[str, np.ndarray], threshold: Optional[float] = 0.5
) -> Dict:
    """ Return prediction results that are above the threshold if any.

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
            or MaskRCNN model, detached in the form of numpy array
        threshold: iou threshold for a positive detection. Note: set
            threshold to None to omit a threshold
    """
    # apply score threshold
    if threshold:
        selected = pred["scores"] > threshold
        pred = {k: v[selected] for k, v in pred.items()}
    # apply mask threshold
    if "masks" in pred:
        pred["masks"] = pred["masks"] > 0.5
    return pred


def _get_pretrained_rcnn(
    model_func: Callable[..., nn.Module],
    # transform parameters
    min_size: int = 800,
    max_size: int = 1333,
    # RPN parameters
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    # Box parameters
    box_score_thresh: int = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        model_func: pretrained R-CNN model generating functions, such as
            fasterrcnn_resnet50_fpn(), get_pretrained_fasterrcnn(), etc.
        min_size: minimum size of the image to be rescaled before feeding it to the backbone
        max_size: maximum size of the image to be rescaled before feeding it to the backbone
        rpn_pre_nms_top_n_train: number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test: number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train: number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test: number of proposals to keep after applying NMS during testing
        rpn_nms_thresh: NMS threshold used for postprocessing the RPN proposals

    Returns
        The pre-trained model
    """
    model = model_func(
        pretrained=True,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )
    return model


def _tune_box_predictor(model: nn.Module, num_classes: int) -> nn.Module:
    """ Tune box predictor in the model. """
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # that has num_classes which is based on the dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _tune_mask_predictor(model: nn.Module, num_classes: int) -> nn.Module:
    """ Tune mask predictor in the model. """
    # get the number of input features of mask predictor from the pretrained model
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features, 256, num_classes
    )
    return model


def get_pretrained_fasterrcnn(num_classes: int = None, **kwargs) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If None, 91 as COCO datasets.

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    """
    # TODO - reconsider that num_classes includes background. This doesn't feel
    #     intuitive.

    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(fasterrcnn_resnet50_fpn, **kwargs)

    # if num_classes is specified, then create new final bounding box
    # prediction layers, otherwise use pre-trained layers
    if num_classes:
        model = _tune_box_predictor(model, num_classes)

    return model


def get_pretrained_maskrcnn(num_classes: int = None, **kwargs) -> nn.Module:
    """ Gets a pretrained Mask R-CNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If None, 91 as COCO datasets.

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(maskrcnn_resnet50_fpn, **kwargs)

    # if num_classes is specified, then create new final bounding box
    # and mask prediction layers, otherwise use pre-trained layers
    if num_classes:
        model = _tune_box_predictor(model, num_classes)
        model = _tune_mask_predictor(model, num_classes)

    return model


def get_pretrained_keypointrcnn(
    num_classes: int = None, num_keypoints: int = None, **kwargs
) -> nn.Module:
    """ Gets a pretrained Keypoint R-CNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If none of num_classes and num_keypoints below are
            not specified, the pretrained model will be returned.
        num_keypoints: number of keypoints
    Returns
        The model to fine-tune/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/keypoint_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(keypointrcnn_resnet50_fpn, **kwargs)

    if num_classes:
        model = _tune_box_predictor(model, num_classes)

    # tune keypoints predictor in the model
    if num_keypoints:
        # get the number of input features of keypoint predictor from the pretrained model
        in_features = (
            model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        )
        # replace the keypoint predictor with a new one
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_features, num_keypoints
        )

    return model


def _calculate_ap(
    e: CocoEvaluator, 
    iou_thres: float = None,
    area_range: str ='all',
    max_detections: int = 100,
    mode: int = 1,
) -> Dict[str, float]:
    """ Calculate the average precision/recall for differnt IoU ranges.

    Args:
        iou_thres: IoU threshold (options: value in [0.5, 0.55, 0.6, ..., 0.95] or None to average over that range)
        area_range: area size range of the target (options: ['all', 'small', 'medium', 'large'])
        max_detections: maximum number of detection frames in a single image (options: [1, 10, 100])
        mode: set to 1 for average precision and otherwise returns average recall
    """
    ap = {}
    for key in e.coco_eval:
        ap[key] = compute_ap(e.coco_eval[key], iouThr=iou_thres, areaRng=area_range, maxDets=max_detections, ap=mode)

    return ap


def _im_eval_detections(
    iou_threshold: float,
    score_threshold: float,
    gt_bboxes: List[DetectionBbox],
    det_bboxes: List[DetectionBbox],
):
    """ Count number of wrong detections and number of missed objects for a single image """
    # Remove all detections with confidence score below a certain threshold
    if score_threshold is not None:
        det_bboxes = [
            bbox for bbox in det_bboxes if bbox.score > score_threshold
        ]

    # Image level statistics.
    # Store (i) if image has at least one missing ground truth; (ii) if image has at least one incorrect detection.
    im_missed_gt = False
    im_wrong_det = False

    # Object level statistics.
    # Store (i) if ground truth objects were found; (ii) if detections are correct.
    found_gts = [False] * len(gt_bboxes)
    correct_dets = [False] * len(det_bboxes)

    # Check if any object was detected in an image
    if len(det_bboxes) == 0:
        if len(gt_bboxes) > 0:
            im_missed_gt = True

    else:
        # loop over ground truth objects and all detections for a given image
        for gt_index, gt_bbox in enumerate(gt_bboxes):
            gt_label = gt_bbox.label_name

            for det_index, det_bbox in enumerate(det_bboxes):
                det_label = det_bbox.label_name
                iou_overlap = bboxes_iou(gt_bbox, det_bbox)

                # mark as good if detection has same label as the ground truth,
                # and if the intersection-over-union area is above a threshold
                if gt_label == det_label and iou_overlap >= iou_threshold:
                    found_gts[gt_index] = True
                    correct_dets[det_index] = True

        # Check if image has at least one wrong detection, or at least one missing ground truth
        im_wrong_det = min(correct_dets) == 0
        if len(gt_bboxes) > 0 and min(found_gts) == 0:
            im_missed_gt = True

    # Count
    obj_missed_gt = len(found_gts) - np.sum(found_gts)
    obj_wrong_det = len(correct_dets) - np.sum(correct_dets)
    return im_wrong_det, im_missed_gt, obj_wrong_det, obj_missed_gt


def ims_eval_detections(
    detections: List[Dict],
    data_ds: Subset,
    detections_neg: List[Dict] = None,
    iou_threshold: float = 0.5,
    score_thresholds: List[float] = np.linspace(0, 1, 51),
):
    """ Count number of wrong detections and number of missed objects for multiple image """
    score_thresholds = [int(f) for f in score_thresholds]

    # get detection bounding boxes and corresponding ground truth for all images
    det_bboxes_list = [d["det_bboxes"] for d in detections]
    gt_bboxes_list = [
        data_ds.dataset.anno_bboxes[d["idx"]] for d in detections
    ]

    # Get counts for test images
    out = [
        [
            _im_eval_detections(
                iou_threshold,
                score_threshold,
                gt_bboxes_list[i],
                det_bboxes_list[i],
            )
            for i in range(len(det_bboxes_list))
        ]
        for score_threshold in score_thresholds
    ]
    out = np.array(out)
    im_wrong_det_counts = np.sum(out[:, :, 0], 1)
    im_missed_gt_counts = np.sum(out[:, :, 1], 1)
    obj_wrong_det_counts = np.sum(out[:, :, 2], 1)
    obj_missed_gt_counts = np.sum(out[:, :, 3], 1)

    # Count how many images have either a wrong detection or a missed ground truth
    im_error_counts = np.sum(np.max(out[:, :, 0:2], 2), 1)

    # Get counts for negative images
    if detections_neg:
        neg_scores = [
            [box.score for box in d["det_bboxes"]] for d in detections_neg
        ]
        neg_scores = [scores for scores in neg_scores if scores != []]
        im_neg_det_counts = [
            np.sum([np.max(scores) > thres for scores in neg_scores])
            for thres in score_thresholds
        ]
        obj_neg_det_counts = [
            np.sum(np.array(list(itertools.chain(*neg_scores))) > thres)
            for thres in score_thresholds
        ]
        assert (
            len(im_neg_det_counts)
            == len(obj_neg_det_counts)
            == len(score_thresholds)
        )

    else:
        im_neg_det_counts = None
        obj_neg_det_counts = None

    assert (
        len(im_error_counts)
        == len(im_wrong_det_counts)
        == len(im_missed_gt_counts)
        == len(obj_missed_gt_counts)
        == len(obj_wrong_det_counts)
        == len(score_thresholds)
    )

    return (
        score_thresholds,
        im_error_counts,
        im_wrong_det_counts,
        im_missed_gt_counts,
        obj_wrong_det_counts,
        obj_missed_gt_counts,
        im_neg_det_counts,
        obj_neg_det_counts,
    )


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(
        self,
        dataset: Dataset = None,
        model: nn.Module = None,
        im_size: int = None,
        device: torch.device = None,
        labels: List[str] = None,
    ):
        """ Initialize leaner object.

        You can only specify an image size `im_size` if `model` is not given.

        Args:
            dataset: the dataset. This class will infer labels if dataset is present.
            model: the nn.Module you wish to use
            im_size: image size for your model
        """
        # if model is None, dataset must not be
        if not model:
            assert dataset is not None

        # not allowed to specify im size if you're providing a model
        if model:
            assert im_size is None

        # if dataset is not None, labels must be (since it is already set in dataset)
        if not dataset:
            assert labels is not None

        # if im_size is not specified, use 500
        if im_size is None:
            im_size = 500

        self.device = device
        if self.device is None:
            self.device = torch_device()

        self.model = model
        self.dataset = dataset
        self.im_size = im_size

        # make sure '__background__' is not included in labels
        if dataset and "labels" in dataset.__dict__:
            self.labels = dataset.labels
        elif labels is not None:
            self.labels = labels
        else:
            raise ValueError("No labels provided in dataset.labels or labels")

        # setup model, default to fasterrcnn
        if self.model is None:
            self.model = get_pretrained_fasterrcnn(
                len(self.labels) + 1,
                min_size=self.im_size,
                max_size=self.im_size,
            )

        self.model.to(self.device)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def fit(
        self,
        epochs: int,
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        print_freq: int = 10,
        step_size: int = None,
        gamma: float = 0.1,
        skip_evaluation: bool = False,
    ) -> None:
        """ The main training loop. """

        if not self.dataset:
            raise Exception("No dataset provided")

        # reduce learning rate every step_size epochs by a factor of gamma (by default) 0.1.
        if step_size is None:
            step_size = int(np.round(epochs / 1.5))

        # construct our optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # store data in these arrays to plot later
        self.losses = []
        self.ap = []
        self.ap_iou_point_5 = []

        # main training loop
        self.epochs = epochs
        for epoch in range(self.epochs):

            # train for one epoch, printing every 10 iterations
            logger = train_one_epoch(
                self.model,
                self.optimizer,
                self.dataset.train_dl,
                self.device,
                epoch,
                print_freq=print_freq,
            )
            self.losses.append(logger.meters["loss"].median)

            # update the learning rate
            self.lr_scheduler.step()

            # evaluate
            if not skip_evaluation:
                e = self.evaluate(dl=self.dataset.test_dl)
                self.ap.append(_calculate_ap(e))
                self.ap_iou_point_5.append(
                    _calculate_ap(e)
                )

    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss from calling `fit` and average precision on the
        test set. """
        fig = plt.figure(figsize=figsize)
        ap = {k: [dic[k] for dic in self.ap] for k in self.ap[0]}

        for i, (k, v) in enumerate(ap.items()):

            ax1 = fig.add_subplot(1, len(ap), i + 1)

            ax1.set_xlim([0, self.epochs - 1])
            ax1.set_xticks(range(0, self.epochs))
            ax1.set_xlabel("epochs")
            ax1.set_ylabel("loss", color="g")
            ax1.plot(self.losses, "g-")

            ax2 = ax1.twinx()
            ax2.set_ylabel(f"AP for {k}", color="b")
            ax2.plot(v, "b-")

        fig.suptitle("Loss and Average Precision (AP) over Epochs")

    def evaluate(self, dl: DataLoader = None) -> CocoEvaluator:
        """ eval code on validation/test set and saves the evaluation results
        in self.results.

        Raises:
            Exception: if both `dl` and `self.dataset` are None.
        """
        if dl is None:
            if not self.dataset:
                raise Exception("No dataset provided for evaluation")
            dl = self.dataset.test_dl
        self.results = evaluate(self.model, dl, device=self.device)
        return self.results

    def predict(
        self,
        im_or_path: Union[np.ndarray, Union[str, Path]],
        threshold: Optional[int] = 0.5,
    ) -> Dict:
        """ Performs inferencing on an image path or image.

        Args:
            im_or_path: the image array which you can get from
                `Image.open(path)` or a image path
            threshold: the threshold to use to calculate whether the object was
                detected. Note: can be set to None to return all detection
                bounding boxes.

        Return a list of DetectionBbox
        """
        if isinstance(im_or_path, (str, Path)):
            im = Image.open(im_or_path)
            im_path = im_or_path
        else:
            im = im_or_path
            im_path = None

        # convert the image to the format required by the model
        transform = transforms.Compose([transforms.ToTensor()])
        im = transform(im)
        if self.device:
            im = im.to(self.device)

        model = self.model.eval()  # eval mode
        with torch.no_grad():
            pred = model([im])[0]

        # detach prediction results to cpu
        pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
        return _extract_od_results(
            _apply_threshold(pred, threshold=threshold), self.labels, im_path
        )

    def predict_dl(
        self, dl: DataLoader, threshold: Optional[float] = 0.5
    ) -> List[DetectionBbox]:
        """ Predict all images in a dataloader object.

        Args:
            dl: the dataloader to predict on
            threshold: iou threshold for a positive detection. Note: set
                threshold to None to omit a threshold

        Returns a list of results
        """
        pred_generator = self.predict_batch(dl, threshold=threshold)
        return [pred for preds in pred_generator for pred in preds]

    def predict_batch(
        self, dl: DataLoader, threshold: Optional[float] = 0.5
    ) -> Generator[List[DetectionBbox], None, None]:
        """ Batch predict

        Args
            dl: A DataLoader to load batches of images from
            threshold: iou threshold for a positive detection. Note: set
                threshold to None to omit a threshold

        Returns an iterator that yields a batch of detection bboxes for each
        image that is scored.
        """

        model = self.model.eval()

        for i, batch in enumerate(dl):
            ims, infos = batch
            ims = [im.to(self.device) for im in ims]
            with torch.no_grad():
                raw_dets = model(ims)

            results = []
            for det, info in zip(raw_dets, infos):
                im_id = int(info["image_id"].item())
                # detach prediction results to cpu
                pred = {k: v.detach().cpu().numpy() for k, v in det.items()}
                extracted_res = _extract_od_results(
                    _apply_threshold(pred, threshold=threshold),
                    self.labels,
                    dl.dataset.dataset.im_paths[im_id],
                )
                results.append({"idx": im_id, **extracted_res})

            yield results

    def save(
        self, name: str, path: str = None, overwrite: bool = True
    ) -> None:
        """ Saves the model

        Save your model in the following format:
        /data_path()
        +-- <name>
        |   +-- meta.json
        |   +-- model.pt

        The meta.json will contain information like the labels and the im_size
        The model.pt will contain the weights of the model

        Args:
            name: the name you wish to save your model under
            path: optional path to save your model to, will use `data_path`
                otherwise
            overwrite: overwrite existing models

        Raise:
            Exception if model file already exists but overwrite is set to
            false

        Returns None
        """
        if path is None:
            path = Path(self.dataset.root) / "models"

        # make dir if not exist
        if not Path(path).exists():
            os.mkdir(path)

        # make dir to contain all model/meta files
        model_path = Path(path) / name
        if model_path.exists():
            if overwrite:
                shutil.rmtree(str(model_path))
            else:
                raise Exception(
                    f"Model of {name} already exists in {path}. Set `overwrite=True` or use another name"
                )
        os.mkdir(model_path)

        # set names
        pt_path = model_path / f"model.pt"
        meta_path = model_path / f"meta.json"

        # save pt
        torch.save(self.model.state_dict(), pt_path)

        # save meta file
        meta_data = {"labels": self.dataset.labels, "im_size": self.im_size}
        with open(meta_path, "w") as meta_file:
            json.dump(meta_data, meta_file)

        print(f"Model is saved to {model_path}")

    def load(self, name: str = None, path: str = None) -> None:
        """ Loads a model.

        Loads a model that is saved in the format that is outputted in the
        `save` function.

        Args:
            name: The name of the model you wish to load. If no name is
            specified, the function will still look for a model under the path
            specified by `data_path`. If multiple models are available in that
            path, it will require you to pass in a name to specify which one to
            use.
            path: Pass in a path if the model is not located in the
            `data_path`. Otherwise it will assume that it is.

        Raise:
            Exception if passed in name/path is invalid and doesn't exist
        """

        # set path
        if not path:
            if self.dataset:
                path = Path(self.dataset.root) / "models"
            else:
                raise Exception("Specify a `path` parameter")

        # if name is given..
        if name:
            model_path = path / name

            pt_path = model_path / "model.pt"
            if not pt_path.exists():
                raise Exception(
                    f"No model file named model.pt exists in {model_path}"
                )

            meta_path = model_path / "meta.json"
            if not meta_path.exists():
                raise Exception(
                    f"No model file named meta.txt exists in {model_path}"
                )

        # if no name is given, we assume there is only one model, otherwise we
        # throw an error
        else:
            models = [f.path for f in os.scandir(path) if f.is_dir()]

            if len(models) == 0:
                raise Exception(f"No model found in {path}.")
            elif len(models) > 1:
                print(
                    f"Multiple models were found in {path}. Please specify which you wish to use in the `name` argument."
                )
                for model in models:
                    print(model)
                exit()
            else:
                pt_path = Path(models[0]) / "model.pt"
                meta_path = Path(models[0]) / "meta.json"

        # load into model
        self.model.load_state_dict(
            torch.load(pt_path, map_location=torch_device())
        )

        # load meta info
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
            self.labels = meta_data["labels"]

    @classmethod
    def from_saved_model(cls, name: str, path: str, mask: bool = False) -> "DetectionLearner":
        """ Create an instance of the DetectionLearner from a saved model.

        This function expects the format that is outputted in the `save`
        function.

        Args:
            name: the name of the model you wish to load
            path: the path to get your model from
            mask: if the model is an instance of maskrcnn

        Returns:
            A DetectionLearner object that can inference.
        """
        path = Path(path)

        meta_path = path / name / "meta.json"
        assert meta_path.exists()

        im_size, labels = None, None
        with open(meta_path) as json_file:
            meta_data = json.load(json_file)
            im_size = meta_data["im_size"]
            labels = meta_data["labels"]

        if mask:
            model = get_pretrained_maskrcnn(
            len(labels) + 1, min_size=im_size, max_size=im_size
            )
        else:
            model = get_pretrained_fasterrcnn(
            len(labels) + 1, min_size=im_size, max_size=im_size
            )

        detection_learner = DetectionLearner(model=model, labels=labels)
        detection_learner.load(name=name, path=path)
        return detection_learner
