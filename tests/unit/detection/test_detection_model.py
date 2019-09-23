# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torchvision.models.detection.faster_rcnn import FasterRCNN
from collections.abc import Iterable
import numpy as np

from utils_cv.detection.bbox import DetectionBbox
from utils_cv.detection.model import (
    get_pretrained_fasterrcnn,
    DetectionLearner,
    _get_det_bboxes,
    _apply_threshold,
    _calculate_ap,
)


def test__get_det_bboxes(od_sample_raw_preds, od_data_path_labels):
    """ test that `_get_det_bboxes` can convert raw preds to DetectionBboxes. """
    det_bboxes = _get_det_bboxes(
        od_sample_raw_preds, labels=od_data_path_labels, im_path=None
    )
    assert type(det_bboxes[0]) == DetectionBbox
    assert len(det_bboxes) == 5


def test__apply_threshold(od_sample_detection_bboxes):
    """ Test `_apply_threshold` and verify it works at different thresholds. """
    det_bboxes = _apply_threshold(od_sample_detection_bboxes, threshold=0.5)
    assert len(det_bboxes) == 3
    det_bboxes = _apply_threshold(od_sample_detection_bboxes, threshold=0.01)
    assert len(det_bboxes) == 5
    det_bboxes = _apply_threshold(od_sample_detection_bboxes, threshold=0.995)
    assert len(det_bboxes) == 2


def test_get_pretrained_fasterrcnn():
    """ Simply test that `get_pretrained_fasterrcnn` returns the correct type for now. """
    assert type(get_pretrained_fasterrcnn(4)) == FasterRCNN


def test__calculate_ap(od_detection_eval):
    """ Test `_calculate_ap`. """
    ret = _calculate_ap(od_detection_eval)
    assert type(ret) == np.float64


def test_detection_learner_init(od_detection_dataset):
    """ Tests detection learner basic init. """
    learner = DetectionLearner(od_detection_dataset)
    assert type(learner) == DetectionLearner


def test_detection_learner_init_model(od_detection_dataset):
    """ Tests detection learner with model settings. """
    classes = len(od_detection_dataset.labels)
    model = get_pretrained_fasterrcnn(
        num_classes=classes, min_size=600, max_size=2000
    )
    learner = DetectionLearner(od_detection_dataset, model=model)
    assert type(learner) == DetectionLearner
    assert learner.model == model
    assert learner.model != get_pretrained_fasterrcnn(classes)


def test_detection_learner_train_one_epoch(od_detection_learner):
    """ Simply test that a small training loop works. """
    od_detection_learner.fit(epochs=1)


def test_detection_learner_plot_precision_loss_curves(od_detection_learner):
    """ Simply test that `plot_precision_loss_curves` works. """
    od_detection_learner.plot_precision_loss_curves()


def test_detection_learner_evalute(od_detection_learner):
    """ Simply test that `evaluate` works. """
    od_detection_learner.evaluate()


def test_detection_learner_predict(od_detection_learner, od_cup_path):
    """ Simply test that `predict` works. """
    bboxes = od_detection_learner.predict(od_cup_path)
    assert type(bboxes) == list


def test_detection_learner_predict_threshold(
    od_detection_learner, od_cup_path
):
    """ Simply test that `predict` works with a threshold by setting a really high threshold. """
    bboxes = od_detection_learner.predict(od_cup_path, threshold=0.9999)
    assert type(bboxes) == list
    assert len(bboxes) == 0


def test_detection_learner_predict_batch(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_batch` works. """
    generator = od_detection_learner.predict_batch(
        od_detection_dataset.test_dl
    )
    assert isinstance(generator, Iterable)


def test_detection_learner_predict_batch_threshold(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_batch` works with a threshold by setting it really high.. """
    generator = od_detection_learner.predict_batch(
        od_detection_dataset.test_dl, threshold=0.9999
    )
    assert isinstance(generator, Iterable)


def test_detection_dataset_predict_dl(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_dl` works. """
    od_detection_learner.predict_dl(od_detection_dataset.test_dl)
