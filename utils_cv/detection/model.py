# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Callable, List, Tuple, Union, Generator, Optional, Dict, Type
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .references.engine import train_one_epoch, evaluate
from .references.coco_eval import CocoEvaluator
from .bbox import _Bbox, DetectionBbox
from ..common.gpu import torch_device
from .data import coco_labels
from .dataset import DetectionDataset


def _get_det_bboxes(
    pred: List[dict], labels: List[str], im_path: str = None
) -> List[DetectionBbox]:
    """ Gets the bounding boxes and labels from the prediction object

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
              model
        labels: list of labels
        im_path: the image path of the preds

    Return:
        a list of DetectionBboxes
    """
    pred_labels = pred[0]["labels"].detach().cpu().numpy().tolist()
    pred_boxes = (
        pred[0]["boxes"].detach().cpu().numpy().astype(np.int32).tolist()
    )
    pred_scores = pred[0]["scores"].detach().cpu().numpy().tolist()

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

    return det_bboxes


def _apply_threshold(
    det_bboxes: List[DetectionBbox], threshold: Optional[float] = 0.5
) -> List[DetectionBbox]:
    """ Filters the list of DetectionBboxes by score threshold. """
    return (
        [det_bbox for det_bbox in det_bboxes if det_bbox.score > threshold]
        if threshold is not None
        else det_bboxes
    )


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
        The model to fine-tine/inference with
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
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # that has num_classes which is based on the dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _tune_mask_predictor(model: nn.Module, num_classes: int) -> nn.Module:
    # get the number of input features of mask predictor from the pretrained model
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features, 256, num_classes)
    return model


def _tune_keypoint_predictor(model: nn.Module, num_keypoints: int) -> nn.Module:
    # get the number of input features of keypoint predictor from the pretrained model
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    # replace the keypoint predictor with a new one
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    return model


def get_pretrained_fasterrcnn(
    num_classes: int = None,
    **kwargs,
) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        num_classes: number of output classes of the model (including the background).

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    """
    # TODO - reconsider that num_classes includes background. This doesn't feel intuitive.

    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(
        fasterrcnn_resnet50_fpn,
        **kwargs,
    )

    if num_classes:
        model = _tune_box_predictor(model, num_classes)

    return model


def get_pretrained_maskrcnn(
    num_classes: int = None,
    **kwargs,
) -> nn.Module:
    """ Gets a pretrained Mask R-CNN model

    Args:
        num_classes: number of output classes of the model (including the background)

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(
        maskrcnn_resnet50_fpn,
        **kwargs,
    )

    if num_classes:
        model = _tune_box_predictor(model, num_classes)
        model = _tune_mask_predictor(model, num_classes)

    return model


def get_pretrained_keypointrcnn(
    num_classes: int = None,
    num_keypoints: int = None,
    **kwargs,
) -> nn.Module:
    """ Gets a pretrained Keypoint R-CNN model

    Args:
        num_classes: number of output classes of the model (including the background)
        num_keypoints: number of keypoints for the specific category
    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/keypoint_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(
        keypointrcnn_resnet50_fpn,
        **kwargs,
    )

    if num_classes:
        model = _tune_box_predictor(model, num_classes)
        model = _tune_mask_predictor(model, num_classes)

    if num_keypoints:
        model = _tune_keypoint_predictor(model, num_keypoints)

    return model


def _calculate_ap(e: CocoEvaluator) -> Dict[str, float]:
    """ Calculate the Average Precision (AP) by averaging all iou
    thresholds across all labels.

    coco_eval.eval['precision'] is a 5-dimensional array. Each dimension
    represents the following:
    1. [T] 10 evenly distributed thresholds for IoU, from 0.5 to 0.95.
    2. [R] 101 recall thresholds, from 0 to 101
    3. [K] label, set to slice(0, None) to get precision over all the labels in
    the dataset. Then take the mean over all labels.
    4. [A] area size range of the target (all-0, small-1, medium-2, large-3)
    5. [M] The maximum number of detection frames in a single image where index
    0 represents max_det=1, 1 represents max_det=10, 2 represents max_det=100

    Therefore, coco_eval.eval['precision'][0, :, 0, 0, 2] represents the value
    of 101 precisions corresponding to 101 recalls from 0 to 100 when IoU=0.5.
    """
    precision_settings = (slice(0, None), slice(0, None), slice(0, None), 0, 2)
    ap = {
        k: np.mean(np.mean(v.eval["precision"][precision_settings]))
        for k, v in e.coco_eval.items()
    }
    return ap


def _get_num_classes(dataset: Optional[DetectionDataset]) -> int:
    return (
        len(dataset.labels) + 1
        if dataset and "labels" in dataset.__dict__
        else len(coco_labels())
    )


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(
        self,
        dataset: DetectionDataset = None,
        model: nn.Module = None,
        device: torch.device = None,
    ):
        """ Initialize leaner object. """
        self.device = device
        if self.device is None:
            self.device = torch_device()

        self.dataset = dataset

        if dataset and "labels" in dataset.__dict__:
            self.labels = ["__background__"] + dataset.labels
        else:
            self.labels = coco_labels()

        # setup model, default to fasterrcnn
        self.model = model
        if self.model is None:
            self.model = (
                get_pretrained_fasterrcnn(len(self.labels))
                if self.dataset
                else get_pretrained_fasterrcnn()
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
    ) -> None:
        """ The main training loop. """

        if not self.dataset:
            return

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
            e = self.evaluate(dl=self.dataset.test_dl)
            self.ap.append(_calculate_ap(e))

    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss from calling `fit` and average precision on the
        test set. """
        fig = plt.figure(figsize=figsize)
        ap = {k: [dic[k] for dic in self.ap] for k in self.ap[0]}

        for i, (k, v) in enumerate(ap.items()):

            ax1 = fig.add_subplot(1, len(ap), i+1)

            ax1.set_xlim([0, self.epochs - 1])
            ax1.set_xticks(range(0, self.epochs))
            ax1.set_title("Loss and Average Precision over epochs")
            ax1.set_xlabel("epochs")
            ax1.set_ylabel("loss", color="g")
            ax1.plot(self.losses, "g-")

            ax2 = ax1.twinx()
            ax2.set_ylabel("average precision for ".format(k), color="b")
            ax2.plot(v, "b-")

    def evaluate(self, dl: DataLoader = None) -> Union[CocoEvaluator, None]:
        """ eval code on validation/test set and saves the evaluation results
        in self.results.
        """
        if dl is None:
            if not self.dataset:
                return
            dl = self.dataset.test_dl
        self.results = evaluate(self.model, dl, device=self.device)
        return self.results

    def _transform(
        self,
        im: Union[str, Path, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """ Convert the image to the format required by the model. """
        transform = transforms.Compose([transforms.ToTensor()])
        im = transform(im)
        if self.device:
            im.to(self.device)
        return im

    @classmethod
    def _apply_threshold(
        cls,
        pred: Dict,
        threshold: Optional[int] = 0.5,
        mask_threshold: Optional[int] = 0.5,
        **kwargs,
    ) -> Dict:
        """ Return prediction results that are above the threshold if any. """
        # detach prediction results to cpu
        pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
        # apply score threshold
        if threshold:
            selected = pred['scores'] > threshold
            pred = {k: v[selected] for k, v in pred.items()}
        if "masks" in pred and mask_threshold:
            pred["masks"] = pred["masks"] > mask_threshold
        return pred

    def _get_det_bboxes(
        self,
        pred: Dict,
        im_path: Union[str, Path]
    ) -> List[Type[_Bbox]]:
        return DetectionBbox.from_arrays(
            pred['boxes'].tolist(),
            score=pred['scores'].tolist(),
            label_idx=pred['labels'].tolist(),
            label_name=np.array(self.labels)[pred['labels']].tolist(),
            im_path=im_path,
        )

    def _process_pred_results(
        self,
        pred: Dict,
        im_path: Union[str, Path]
    ) -> Union[List[Type[_Bbox]], Tuple]:

        res = self._get_det_bboxes(pred, im_path)
        if "masks" in pred:
            res = (res, pred["masks"].squeeze())
        elif "keypoints" in pred:
            res = (res, pred["keypoints"])
        return res

    @classmethod
    def _pack_pred_results(cls, res: List, infos: Dict) -> List[Dict]:
        if not isinstance(res[0], tuple):
            return [
                {
                    "idx": t["image_id"],
                    "det_bboxes": r
                } for r, t in zip(res, infos)
            ]
        elif res[0][1].shape[2] != 3:
            return [
                {
                    "idx": t["image_id"],
                    "det_bboxes": b,
                    "masks": m,
                } for (b, m), t in zip(res, infos)
            ]
        else:
            return [
                {
                    "idx": t["image_id"],
                    "det_bboxes": b,
                    "keypoints": k,
                } for (b, k), t in zip(res, infos)
            ]

    def predict(
        self,
        im_or_path: Union[np.ndarray, Union[str, Path]],
        threshold: Optional[int] = 0.5,
        **kwargs,
    ) -> List[Type[_Bbox]]:
        """ Performs inferencing on an image path or image.

        Args:
            im_or_path: the image array which you can get from `Image.open(path)` OR a
            image path
            threshold: the threshold to use to calculate whether the object was
            detected. Note: can be set to None to return all detection bounding
            boxes.

        Return a list of DetectionBbox
        """
        im, im_path = (
            (Image.open(im_or_path), im_or_path)
            if isinstance(im_or_path, (str, Path))
            else (im_or_path, None)
        )
        im = self._transform(im)
        model = self.model.eval()  # eval mode
        with torch.no_grad():
            pred = model([im])

        pred = [
            self._apply_threshold(p, threshold, **kwargs) for p in pred
        ]
        return self._process_pred_results(pred[0], im_path)

    def predict_dl(
        self, dl: DataLoader, threshold: Optional[float] = 0.5,
    ) -> List[DetectionBbox]:
        """ Predict all images in a dataloader object.

        Args:
            dl: the dataloader to predict on
            threshold: iou threshold for a positive detection. Note: set
            threshold to None to omit a threshold

        Returns a list of results
        """
        pred_generator = self.predict_batch(dl, threshold=threshold)
        res = [pred for preds in pred_generator for pred in preds]
        return res

    def predict_batch(
        self, dl: DataLoader, threshold: Optional[float] = 0.5, **kwargs,
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

            raw_dets = [
                self._apply_threshold(p, threshold, **kwargs) for p in
                raw_dets
            ]
            res = [
                self._process_pred_results(
                    p,
                    dl.dataset.dataset.im_paths[int(t["image_id"].item())]
                ) for p, t in zip(raw_dets, infos)
            ]
            yield DetectionLearner._pack_pred_results(res, infos)
