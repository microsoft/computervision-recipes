# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from collections.abc import Iterable
import numpy as np
import pytest
import shutil
from pathlib import Path
from typing import Tuple

from utils_cv.detection.bbox import DetectionBbox
from utils_cv.detection.model import (
    get_pretrained_fasterrcnn,
    get_pretrained_keypointrcnn,
    get_pretrained_maskrcnn,
    DetectionLearner,
    _apply_threshold,
    _calculate_ap,
    _extract_od_results,
    ims_eval_detections,
)


def test__extract_od_results(od_sample_raw_preds, od_data_path_labels):
    """ test that `_extract_od_results` can convert raw preds. """
    pred = {
        k: v.detach().cpu().numpy() for k, v in od_sample_raw_preds[0].items()
    }
    out = _extract_od_results(pred, labels=od_data_path_labels, im_path=None)
    bboxes = out["det_bboxes"]
    assert type(bboxes[0]) == DetectionBbox
    assert len(bboxes) == 5
    assert out["masks"].shape == (5, 666, 499)
    assert out["keypoints"].shape == (5, 13, 3)


def test__apply_threshold(od_sample_raw_preds):
    """ Test `_apply_threshold` and verify it works at different thresholds. """
    # test cases: [(threshold, num, mask_pixels)]
    test_cases = [
        (0.5, 3, (21146, 28098, 28458)),
        (0.01, 5, (21146, 28098, 28458, 28356, 21311)),
        (0.995, 2, (21146, 28098)),
    ]
    out = {
        k: v.detach().cpu().numpy() for k, v in od_sample_raw_preds[0].items()
    }
    for threshold, num, mask_pixels in test_cases:
        pred = _apply_threshold(out, threshold=threshold)
        for v in pred.values():
            assert len(v) == num
        for mask, num_pixels in zip(pred["masks"], mask_pixels):
            assert np.sum(mask) == num_pixels


def test_get_pretrained_fasterrcnn():
    """ Simply test that `get_pretrained_fasterrcnn` returns the correct type for now. """
    assert type(get_pretrained_fasterrcnn(4)) == FasterRCNN


def test_get_pretrained_maskrcnn():
    """ Simply test that `get_pretrained_maskrcnn` returns the correct type for now. """
    assert type(get_pretrained_maskrcnn(4)) == MaskRCNN


def test_get_pretrained_keypointrcnn():
    """ Simply test that `get_pretrained_keypointrcnn` returns the correct type for now. """
    assert type(get_pretrained_keypointrcnn(2, 6)) == KeypointRCNN


@pytest.mark.gpu
def test__calculate_ap(od_detection_eval):
    """ Test `_calculate_ap`. """
    ret = _calculate_ap(od_detection_eval)
    assert type(ret) == dict
    for v in ret.values():
        assert type(v) == np.float64


@pytest.mark.gpu
def test_ims_eval_detections(od_detection_dataset, od_detections):
    """ Test `ims_eval_detections`. """
    (
        score_thresholds,
        im_error_counts,
        im_wrong_det_counts,
        im_missed_gt_counts,
        obj_wrong_det_counts,
        obj_missed_gt_counts,
        im_neg_det_counts,
        obj_neg_det_counts,
    ) = ims_eval_detections(
        od_detections, od_detection_dataset.test_ds, od_detections
    )
    assert len(score_thresholds) == 51
    assert max(im_error_counts) == len(od_detection_dataset.test_ds)
    assert max(im_wrong_det_counts) <= len(od_detection_dataset.test_ds)
    assert max(im_missed_gt_counts) <= len(od_detection_dataset.test_ds)
    assert max(im_neg_det_counts) <= len(od_detection_dataset.test_ds)
    for i in range(len(score_thresholds) - 1):
        assert im_wrong_det_counts[i] >= im_wrong_det_counts[i + 1]
        assert im_missed_gt_counts[i] <= im_missed_gt_counts[i + 1]
        assert obj_wrong_det_counts[i] >= obj_wrong_det_counts[i + 1]
        assert obj_missed_gt_counts[i] <= obj_missed_gt_counts[i + 1]
        assert im_neg_det_counts[i] >= im_neg_det_counts[i + 1]
        assert obj_neg_det_counts[i] >= obj_neg_det_counts[i + 1]


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


@pytest.mark.gpu
def test_detection_learner_train_one_epoch(od_detection_learner,):
    """ Simply test that a small training loop works. """
    od_detection_learner.fit(epochs=1)


@pytest.mark.gpu
def test_detection_mask_learner_train_one_epoch(od_detection_mask_learner,):
    """ Simply test that a small training loop works for mask learner. """
    od_detection_mask_learner.fit(epochs=1)


@pytest.mark.gpu
def test_detection_keypoint_learner_train_one_epoch(
    od_detection_keypoint_learner,
):
    """ Simply test that a small training loop works for keypoint learner. """
    od_detection_keypoint_learner.fit(epochs=1, skip_evaluation=True)


@pytest.mark.gpu
def test_detection_learner_plot_precision_loss_curves(od_detection_learner,):
    """ Simply test that `plot_precision_loss_curves` works. """
    od_detection_learner.plot_precision_loss_curves()


@pytest.mark.gpu
def test_detection_mask_learner_plot_precision_loss_curves(
    od_detection_mask_learner,
):
    """ Simply test that `plot_precision_loss_curves` works for mask learner. """
    od_detection_mask_learner.plot_precision_loss_curves()


@pytest.mark.gpu
def test_detection_learner_evaluate(od_detection_learner,):
    """ Simply test that `evaluate` works. """
    od_detection_learner.evaluate()


@pytest.mark.gpu
def test_detection_mask_learner_evaluate(od_detection_mask_learner,):
    """ Simply test that `evaluate` works for mask learner. """
    od_detection_mask_learner.evaluate()


@pytest.mark.gpu
def test_detection_learner_predict(od_detection_learner, od_cup_path):
    """ Simply test that `predict` works. """
    bboxes = od_detection_learner.predict(od_cup_path)["det_bboxes"]
    assert type(bboxes) == list


@pytest.mark.gpu
def test_detection_mask_learner_predict(
    od_detection_mask_learner, od_cup_path
):
    """ Simply test that `predict` works for mask learner. """
    pred = od_detection_mask_learner.predict(od_cup_path, threshold=0.1)
    bboxes = pred["det_bboxes"]
    masks = pred["masks"]
    assert type(bboxes) == list
    assert type(masks) == np.ndarray
    assert len(bboxes) == len(masks)


@pytest.mark.gpu
def test_detection_keypoint_learner_predict(
    od_detection_keypoint_learner, od_cup_path
):
    """ Simply test that `predict` works for keypoint learner. """
    pred = od_detection_keypoint_learner.predict(od_cup_path, threshold=0.1)
    bboxes = pred["det_bboxes"]
    keypoints = pred["keypoints"]
    assert type(bboxes) == list
    assert type(keypoints) == np.ndarray
    assert len(bboxes) == len(keypoints)


@pytest.mark.gpu
def test_detection_learner_predict_threshold(
    od_detection_learner, od_cup_path
):
    """ Simply test that `predict` works with a threshold by setting a really
    high threshold.
    """
    bboxes = od_detection_learner.predict(od_cup_path, threshold=0.9999)[
        "det_bboxes"
    ]
    assert type(bboxes) == list
    assert len(bboxes) == 0


@pytest.mark.gpu
def test_detection_mask_learner_predict_threshold(
    od_detection_mask_learner, od_cup_path
):
    """ Simply test that `predict` works for mask learner with a threshold by
    setting a really high threshold.
    """
    pred = od_detection_mask_learner.predict(od_cup_path, threshold=0.9999)
    bboxes = pred["det_bboxes"]
    masks = pred["masks"]
    assert type(bboxes) == list
    assert type(masks) == np.ndarray
    assert len(bboxes) == len(masks)
    assert len(bboxes) == 0


@pytest.mark.gpu
def test_detection_keypoint_learner_predict_threshold(
    od_detection_keypoint_learner, od_cup_path
):
    """ Simply test that `predict` works for keypoint learner with a threshold by
    setting a really high threshold.
    """
    pred = od_detection_keypoint_learner.predict(od_cup_path, threshold=0.9999)
    bboxes = pred["det_bboxes"]
    keypoints = pred["keypoints"]
    assert type(bboxes) == list
    assert type(keypoints) == np.ndarray
    assert len(bboxes) == len(keypoints)
    assert len(bboxes) == 0


@pytest.mark.gpu
def test_detection_learner_predict_batch(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_batch` works. """
    generator = od_detection_learner.predict_batch(
        od_detection_dataset.test_dl
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_mask_learner_predict_batch(
    od_detection_mask_learner, od_detection_mask_dataset
):
    """ Simply test that `predict_batch` works for mask learner. """
    generator = od_detection_mask_learner.predict_batch(
        od_detection_mask_dataset.test_dl
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_keypoint_learner_predict_batch(
    od_detection_keypoint_learner, tiny_od_detection_keypoint_dataset
):
    """ Simply test that `predict_batch` works for keypoint learner. """
    generator = od_detection_keypoint_learner.predict_batch(
        tiny_od_detection_keypoint_dataset.test_dl
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_learner_predict_batch_threshold(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_batch` works with a threshold by setting it
    really high.
    """
    generator = od_detection_learner.predict_batch(
        od_detection_dataset.test_dl, threshold=0.9999
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_mask_learner_predict_batch_threshold(
    od_detection_mask_learner, od_detection_mask_dataset
):
    """ Simply test that `predict_batch` works for mask learner with a
    threshold by setting it really high.
    """
    generator = od_detection_mask_learner.predict_batch(
        od_detection_mask_dataset.test_dl, threshold=0.9999
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_keypoint_learner_predict_batch_threshold(
    od_detection_keypoint_learner, tiny_od_detection_keypoint_dataset
):
    """ Simply test that `predict_batch` works for keypoint learner with a
    threshold by setting it really high.
    """
    generator = od_detection_keypoint_learner.predict_batch(
        tiny_od_detection_keypoint_dataset.test_dl, threshold=0.9999
    )
    assert isinstance(generator, Iterable)


@pytest.mark.gpu
def test_detection_dataset_predict_dl(
    od_detection_learner, od_detection_dataset
):
    """ Simply test that `predict_dl` works. """
    od_detection_learner.predict_dl(od_detection_dataset.test_dl)


@pytest.mark.gpu
def test_detection_mask_dataset_predict_dl(
    od_detection_mask_learner, od_detection_mask_dataset
):
    """ Simply test that `predict_dl` works for mask learner. """
    od_detection_mask_learner.predict_dl(od_detection_mask_dataset.test_dl)


@pytest.mark.gpu
def test_detection_keypoint_dataset_predict_dl(
    od_detection_keypoint_learner, tiny_od_detection_keypoint_dataset
):
    """ Simply test that `predict_dl` works for mask learner. """
    od_detection_keypoint_learner.predict_dl(
        tiny_od_detection_keypoint_dataset.test_dl
    )


def validate_saved_model(name: str, path: str) -> None:
    """ Tests that saved model is there """
    assert (Path(path)).exists()
    assert (Path(path) / name).exists()
    assert (Path(path) / name / "meta.json").exists()
    assert (Path(path) / name / "model.pt").exists()


@pytest.mark.gpu
def test_detection_save_model(od_detection_learner, tiny_od_data_path):
    """ Test that save function works. """

    # test without path (default to using data_path()/models)
    model_name = "my_test_model"
    od_detection_learner.save(model_name)
    validate_saved_model(model_name, Path(tiny_od_data_path) / "models")

    # test with path
    od_detection_learner.save(
        model_name, str(Path(tiny_od_data_path) / "layer")
    )
    validate_saved_model(model_name, str(Path(tiny_od_data_path) / "layer"))

    # test with overwrite
    with pytest.raises(Exception):
        od_detection_learner.save(model_name, overwrite=False)


@pytest.mark.gpu
@pytest.fixture(scope="session")
def saved_model(od_detection_learner, tiny_od_data_path) -> Tuple[str, Path]:
    """ A saved model so that loading functions can reuse. """
    model_name = "test_fixture_model"
    od_detection_learner.save(model_name)
    assert (Path(tiny_od_data_path) / "models" / model_name).exists()
    return model_name, Path(tiny_od_data_path) / "models"


@pytest.mark.gpu
def test_detection_load_model(
    od_detection_learner, tiny_od_data_path, saved_model
):
    """ Test that load function works. """

    # test basic loading
    name, path = saved_model
    od_detection_learner.load(name=name)
    od_detection_learner.load(name=name, path=path)
    assert od_detection_learner.labels is not None

    # do not specify name or path, it should quietly exit
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        od_detection_learner.load()
    assert pytest_wrapped_e.type == SystemExit

    # test of no model files exists
    shutil.rmtree(path / name)
    shutil.rmtree(Path(tiny_od_data_path) / "models")

    with pytest.raises(Exception):
        od_detection_learner.load()

    # test if only one model file exists in the `data_path`
    od_detection_learner.save("test_fixture_model")
    od_detection_learner.load()


@pytest.mark.gpu
def test_detection_init_from_saved_model(saved_model):
    """ Test that we can create an detection learner from a saved model. """
    name, path = saved_model
    DetectionLearner.from_saved_model(name, path)
