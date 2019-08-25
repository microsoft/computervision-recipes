# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Tuple, Union, Generator, Optional
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from .references.engine import train_one_epoch, evaluate
from .references.coco_eval import CocoEvaluator
from .bbox import DetectionBbox
from ..common.gpu import torch_device


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
    pred_labels = list(pred[0]["labels"].cpu().numpy())
    pred_boxes = list(pred[0]["boxes"].detach().cpu().numpy().astype(np.int32))
    pred_scores = list(pred[0]["scores"].detach().cpu().numpy())

    det_bboxes = []
    for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
        label_name = labels[label]
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


def get_pretrained_fasterrcnn(num_classes: int) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        num_classes: the number of classes to be detected

    Returns
        The model to fine-tine/inference with
    """
    # load a model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # that has num_classes which is based on the dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _calculate_ap(e: CocoEvaluator) -> float:
    """ Calculate the Average Precision (AP) by averaging all iou
    thresholds across all labels.

    see `utils.detection.plot:_get_precision_recall_settings` to
    get information on the precision_setting variable below and what the
    indicies mean.
    """
    precision_settings = (slice(0, None), slice(0, None), slice(0, None), 0, 2)
    coco_eval = e.coco_eval["bbox"].eval["precision"]
    return np.mean(np.mean(coco_eval[precision_settings]))


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(self, dataset: Dataset, model: nn.Module = None):
        """ Initialize leaner object. """
        self.device = torch_device()
        self.model = model
        self.dataset = dataset

        # setup model, default to fasterrcnn
        if self.model is None:
            self.model = get_pretrained_fasterrcnn(len(dataset.labels)).to(
                self.device
            )

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
    ) -> None:
        """ The main training loop. """

        # construct our optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
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
            e = self.evaluate(dl=self.dataset.train_dl)
            self.ap.append(_calculate_ap(e))

    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss from calling `fit` and average precision on the
        test set. """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        ax1.set_xlim([0, self.epochs - 1])
        ax1.set_xticks(range(0, self.epochs))
        ax1.set_title("Loss and Average Precision over epochs")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss", color="g")
        ax1.plot(self.losses, "g-")

        ax2 = ax1.twinx()
        ax2.set_ylabel("average precision", color="b")
        ax2.plot(self.ap, "b-")

    def evaluate(self, dl: DataLoader = None) -> CocoEvaluator:
        """ eval code on validation/test set and saves the evaluation results in self.results. """
        if dl is None:
            dl = self.dataset.test_dl
        self.results = evaluate(self.model, dl, device=self.device)
        return self.results

    def predict(
        self,
        im_or_path: Union[np.ndarray, Union[str, Path]],
        threshold: Optional[int] = 0.6,
    ) -> List[DetectionBbox]:
        """ Performs inferencing on an image path or image.

        Args:
            im_or_path: the image array which you can get from `Image.open(path)` OR a
            image path
            threshold: the threshold to use to calculate whether the object was
            detected. Note: can be set to None to return all detection bounding
            boxes.

        Raises:
            TypeError is the im object is a path or str to the image instead of
            an nd.array

        Return a list of DetectionBbox
        """
        im = (
            Image.open(im_or_path)
            if isinstance(im_or_path, (str, Path))
            else im_or_path
        )

        transform = transforms.Compose([transforms.ToTensor()])
        im = transform(im).cuda()
        model = self.model.eval()  # eval mode
        with torch.no_grad():
            pred = model([im])
        labels = self.dataset.labels
        det_bboxes = _get_det_bboxes(pred, labels=labels)

        # limit to threshold if threshold is set
        return _apply_threshold(det_bboxes, threshold)

    def predict_dl(
        self, dl: DataLoader, threshold: Optional[float] = 0.5
    ) -> List[DetectionBbox]:
        """ Predict all images in a dataloader object.

        Args:
            dl: the dataloader to predict on
            threshold: iou threshold for a positive detection. Note: set
            threshold to None to omit a threshold

        Returns a list of DetectionBbox
        """
        pred_generator = self.predict_batch(dl, threshold=threshold)
        det_bboxes = [pred for preds in pred_generator for pred in preds]
        return det_bboxes

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

        labels = self.dataset.labels
        model = self.model.eval()

        for i, batch in enumerate(dl):
            ims, infos = batch
            ims = [im.cuda() for im in ims]
            with torch.no_grad():
                raw_dets = model(list(ims))

            det_bbox_batch = []
            for raw_det, info in zip(raw_dets, infos):

                im_idx = int(info["image_id"].numpy())
                im_path = dl.dataset.dataset.get_path_from_idx(im_idx)

                det_bboxes = _get_det_bboxes(
                    [raw_det], labels=labels, im_path=im_path
                )

                det_bboxes = _apply_threshold(det_bboxes, threshold)
                det_bbox_batch.append(
                    {"idx": im_idx, "det_bboxes": det_bboxes}
                )
            yield det_bbox_batch
