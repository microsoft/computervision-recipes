# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from fastai.metrics import accuracy, error_rate
from fastai.vision import cnn_learner, models
from fastai.vision import ImageList, imagenet_stats
from utils_cv.classification.model import (
    TrainMetricsRecorder,
    model_to_learner,
)


def test_model_to_learner():
    # Test if the function loads an ImageNet model (ResNet) trainer
    learn = model_to_learner(models.resnet34(pretrained=True))
    assert len(learn.data.classes) == 1000  # Check Image net classes
    assert isinstance(learn.model, models.ResNet)

    # Test with SqueezeNet
    learn = model_to_learner(models.squeezenet1_0())
    assert len(learn.data.classes) == 1000
    assert isinstance(learn.model, models.SqueezeNet)


@pytest.fixture
def tiny_ic_data(tiny_ic_data_path):
    """ Returns tiny ic data bunch """
    return (
        ImageList.from_folder(tiny_ic_data_path)
        .split_by_rand_pct(valid_pct=0.2, seed=10)
        .label_from_folder()
        .transform(size=299)
        .databunch(bs=16)
        .normalize(imagenet_stats)
    )


def test_train_metrics_recorder(tiny_ic_data):
    model = models.resnet18
    lr = 1e-4
    epochs = 2

    def test_callback(learn):
        tmr = TrainMetricsRecorder(learn)
        learn.callbacks.append(tmr)
        learn.unfreeze()
        learn.fit(epochs, lr)
        return tmr

    # multiple metrics
    learn = cnn_learner(tiny_ic_data, model, metrics=[accuracy, error_rate])
    cb = test_callback(learn)
    assert len(cb.train_metrics) == len(cb.valid_metrics) == epochs
    assert (
        len(cb.train_metrics[0]) == len(cb.valid_metrics[0]) == 2
    )  # we used 2 metrics

    # no metrics
    learn = cnn_learner(tiny_ic_data, model)
    cb = test_callback(learn)
    assert len(cb.train_metrics) == len(cb.valid_metrics) == 0  # no metrics

    # no validation set
    learn = cnn_learner(tiny_ic_data, model, metrics=accuracy)
    learn.data.valid_dl = None
    cb = test_callback(learn)
    assert len(cb.train_metrics) == epochs
    assert len(cb.train_metrics[0]) == 1  # we used 1 metrics
    assert len(cb.valid_metrics) == 0  # no validation
