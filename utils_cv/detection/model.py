# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import List, Tuple, Union, Generator, Optional
from pathlib import Path
import json
import shutil

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


def get_pretrained_fasterrcnn(
    num_classes: int,
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
        num_classes: number of output classes of the model (including the background).
        min_size: minimum size of the image to be rescaled before feeding it to the backbone
        max_size: maximum size of the image to be rescaled before feeding it to the backbone
        rpn_pre_nms_top_n_train: number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test: number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train: number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test: number of proposals to keep after applying NMS during testing
        rpn_nms_thresh: NMS threshold used for postprocessing the RPN proposals
        box_score_thresh: during inference, only return proposals with a classification score greater than box_score_thresh
        box_nms_thresh: NMS threshold for the prediction head. Used during inference
        box_detections_per_img: maximum number of detections per image, for all classes

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py

    """
    # TODO - reconsider that num_classes includes background. This doesn't feel intuitive.

    # load a model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(
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

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # that has num_classes which is based on the dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _calculate_ap(e: CocoEvaluator) -> float:
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
    coco_eval = e.coco_eval["bbox"].eval["precision"]
    return np.mean(np.mean(coco_eval[precision_settings]))


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(
        self,
        dataset: Dataset = None,
        model: nn.Module = None,
        im_size: int = None,
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

        # if im_size is not specified, use 500
        if im_size is None:
            im_size = 500

        self.device = torch_device()
        self.model = model
        self.dataset = dataset
        self.im_size = im_size

        # setup model, default to fasterrcnn
        if self.model is None:
            self.model = get_pretrained_fasterrcnn(
                len(self.dataset.labels) + 1,
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
    ) -> None:
        """ The main training loop. """

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
        threshold: Optional[int] = 0.5,
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

        labels = self.dataset.labels if self.dataset else self.labels
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
                im_path = dl.dataset.dataset.im_paths[im_idx]

                det_bboxes = _get_det_bboxes(
                    [raw_det], labels=labels, im_path=im_path
                )

                det_bboxes = _apply_threshold(det_bboxes, threshold)
                det_bbox_batch.append(
                    {"idx": im_idx, "det_bboxes": det_bboxes}
                )
            yield det_bbox_batch

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
            overwrite: overwite existing models

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
    def from_saved_model(cls, name: str, path: str) -> "DetectionLearner":
        """ Create an instance of the DetectionLearner from a saved model.

        This function expects the format that is outputted in the `save`
        function.

        Args:
            name: the name of the model you wish to load
            path: the path to get your model from

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

        model = get_pretrained_fasterrcnn(
            len(labels) + 1, min_size=im_size, max_size=im_size
        )
        detection_learner = DetectionLearner(model=model)
        detection_learner.load(name=name, path=path)
        return detection_learner
