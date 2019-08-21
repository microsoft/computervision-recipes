# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Tuple, Union, Generator
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
from .helper import Annotation, Bbox
from ..common.gpu import torch_device


def get_bounding_boxes(
    pred: List[dict], categories: List[str] = None, threshold: int = 0.6
) -> Tuple[List[int], List[List[int]]]:
    """ Gets the bounding boxes and labels from a prediction given a threshold.

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
        model
        categories: list of categories
        threshold: the minimum threshold to accept.

    Return:
        a list of labels and bounding boxes that pass the minimum threshold.
    """
    pred_labels = list(pred[0]["labels"].cpu().numpy())
    pred_boxes = list(pred[0]["boxes"].detach().cpu().numpy().astype(np.int32))
    pred_scores = list(pred[0]["scores"].detach().cpu().numpy())

    annos = []
    for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
        if score > threshold:
            bbox = Bbox.from_array(box)
            category_name = (
                categories[label] if categories is not None else None
            )
            anno = Annotation(
                bbox, category_name=category_name, category_idx=label
            )
            annos.append(anno)

    return annos


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


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(
        self,
        dataset: Dataset,
        lr: float,
        model: nn.Module = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
    ):
        """ Initialize leaner object. """
        self.device = torch_device()
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # get training dataloaders and datasets
        self.dataset = dataset
        self.train_ds = dataset.train_ds
        self.test_ds = dataset.test_ds
        self.train_dl = dataset.train_dl
        self.test_dl = dataset.test_dl

        # setup model, default to fasterrcnn
        if self.model is None:
            self.model = get_pretrained_fasterrcnn(len(dataset.categories)).to(
                self.device
            )

        # construct our optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def fit(self, epochs: int, print_freq: int = 10) -> None:
        """ The main training loop. """
        self.losses = []
        self.epochs = epochs
        for epoch in range(self.epochs):

            # train for one epoch, printing every 10 iterations
            logger = train_one_epoch(
                self.model,
                self.optimizer,
                self.train_dl,
                self.device,
                epoch,
                print_freq=print_freq,
            )
            self.losses.append(logger.meters["loss"].median)

            # update the learning rate
            self.lr_scheduler.step()

            # evaluate
            self.evaluate(dl=self.train_dl)

    def plot_losses(self, figsize: Tuple[int, int] = (10, 5)) -> None:
        """ Plot training loss from fitting. """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.epochs - 1])
        ax.set_xticks(range(0, self.epochs))
        ax.set_title("Loss over epochs")
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")
        ax.plot(self.losses)

    def evaluate(self, dl: DataLoader = None) -> CocoEvaluator:
        """ eval code on validation/test set and saves the evaluation results in self.results. """
        if dl is None:
            dl = self.test_dl
        self.results = evaluate(self.model, dl, device=self.device)
        return self.results

    def predict(
        self,
        im_or_path: Union[np.ndarray, Union[str, Path]],
        threshold: int = 0.6,
    ) -> List[Annotation]:
        """ Performs inferencing on an image path or image.

        Args:
            im_or_path: the image array which you can get from `Image.open(path)` OR a
            image path
            threshold: the threshold to use to calculate whether the object was
            detected

        Raises:
            TypeError is the im object is a path or str to the image instead of
            an nd.array

        Return the prediction dictionary object
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
        categories = self.train_ds.dataset.categories
        return get_bounding_boxes(
            pred, categories=categories, threshold=threshold
        )

    def pred_batch(
        self, dl: DataLoader, threshold: int = 0.6
    ) -> Generator[List[Annotation], None, None]:
        """ Batch predict

        Args
            dataset_iterator: takes in a dataloader iterator, and predicts on the batch_size
            specified by it. This can be created by `iter(dataloader)`
            threshold: iou threshold for a positive detection

        Returns an iterator that yields a list of annotations for each image that is scored
        """

        categories = self.dataset.categories
        model = self.model.eval()

        for i, batch in enumerate(dl):
            ims, infos = batch
            ims = [im.cuda() for im in ims]
            with torch.no_grad():
                preds = model(list(ims))

            anno_batch = []
            for pred, info in zip(preds, infos):
                anno = get_bounding_boxes(
                    [pred], categories=categories, threshold=threshold
                )
                anno_batch.append(
                    {"idx": int(info["image_id"].numpy()), "anno": anno}
                )
            yield anno_batch
