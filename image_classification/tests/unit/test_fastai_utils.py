from fastai.metrics import accuracy, error_rate
from fastai.vision import cnn_learner, models

from utils_ic.fastai_utils import TrainMetricsRecorder


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
    assert len(cb.train_metrics[0]) == len(cb.valid_metrics[0]) == 2  # we used 2 metrics

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
