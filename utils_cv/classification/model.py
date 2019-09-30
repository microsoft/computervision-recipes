# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from time import time
from typing import Any, Callable, List, Optional

import fastai.basic_train
from fastai.basic_train import _loss_func2activ, LearnerCallback
from fastai.torch_core import TensorOrNumList
from fastai.basic_data import DatasetType
from fastai.vision import (
    CallbackHandler,
    DataLoader,
    Learner,
    nn,
    ImageDataBunch,
    imagenet_stats,
    PBar,
)
from fastprogress.fastprogress import format_time
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch import Tensor
import numpy as np

from utils_cv.classification.data import imagenet_labels

# Default ImageNet models image size
IMAGENET_IM_SIZE = 224


def hamming_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = 0.2,
    sigmoid: bool = False,
) -> Tensor:
    """ Callback for using hamming accuracy as a evaluation metric.

    Hamming accuracy is one minus the fraction of wrong labels to the total
    number of labels.

    Args:
        y_pred: prediction output
        y_true: true class labels
        threshold: the threshold to consider a positive classification
        sigmoid: whether to apply the sigmoid activation

    Returns:
        The hamming accuracy function as a tensor of dtype float
    """
    if sigmoid:
        y_pred = y_pred.sigmoid()
    if threshold:
        y_pred = y_pred > threshold
    return 1 - (
        (y_pred.float() != y_true).sum() / torch.ones(y_pred.shape).sum()
    )


def zero_one_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = 0.2,
    sigmoid: bool = False,
) -> Tensor:
    """ Callback for using zero-one accuracy as a evaluation metric.

    The zero-one accuracy will classify an entire set of labels for a given
    sample incorrect if it does not entirely match the true set of labels.

    Args:
        y_pred: prediction output
        y_true: true class labels
        threshold: the threshold to consider a positive classification
        sigmoid: whether to apply the sigmoid activation

    Returns:
        The zero-one accuracy function as a tensor with dtype float
    """
    if sigmoid:
        y_pred = y_pred.sigmoid()
    if threshold:
        y_pred = y_pred > threshold

    zero_one_preds = (y_pred.float() != y_true).sum(dim=1)
    zero_one_preds[zero_one_preds >= 1] = 1
    num_labels = y_pred.shape[-1]
    return 1 - (
        zero_one_preds.sum().float() / len(y_pred.reshape(-1, num_labels))
    )


def get_optimal_threshold(
    metric_function: Callable[[Tensor, Tensor, float], Tensor],
    y_pred: Tensor,
    y_true: Tensor,
    thresholds: List[float] = np.linspace(0, 1, 21),
) -> float:
    """ Gets the best threshold to use for the provided metric function.

    This method samples the metric function at evenly distributed threshold
    intervals to find the best threshold.

    Args:
        metric_function: The metric function
        y_pred: predicted probabilities.
        y_true: True class indices.
        samples: The number of samples.

    Returns:
        The threshold that optimizes the metric function.
    """
    optimal_threshold = None
    metric_max = -np.inf
    for threshold in thresholds:
        metric = metric_function(y_pred, y_true, threshold=threshold)
        if metric > metric_max:
            metric_max = metric
            optimal_threshold = threshold
    return optimal_threshold


def model_to_learner(
    model: nn.Module, im_size: int = IMAGENET_IM_SIZE
) -> Learner:
    """Create Learner based on pyTorch ImageNet model.

    Args:
        model (nn.Module): Base ImageNet model. E.g. models.resnet18()
        im_size (int): Image size the model will expect to have.

    Returns:
         Learner: a model trainer for prediction
    """

    # Currently, fast.ai api requires to pass a DataBunch to create a model trainer (learner).
    # To use the learner for prediction tasks without retraining, we have to pass an empty DataBunch.
    # single_from_classes is deprecated, but this is the easiest go-around method.
    # Create ImageNet data spec as an empty DataBunch.
    # Related thread: https://forums.fast.ai/t/solved-get-prediction-from-the-imagenet-model-without-creating-a-databunch/36197/5
    empty_data = ImageDataBunch.single_from_classes(
        "", classes=imagenet_labels(), size=im_size
    ).normalize(imagenet_stats)

    return Learner(empty_data, model)


def get_preds(
    learn: Learner,
    dl: DataLoader,
    with_loss: bool = False,
    n_batch: Optional[int] = None,
    pbar: Optional[PBar] = None,
) -> List[Tensor]:
    """Return predictions and targets on `dl` dataset.
    This function is the same as fastai's Learner.get_preds except this allows an external DataLoader.
    For more details about Learner.get_preds, see:
    https://github.com/fastai/fastai/blob/master/fastai/basic_train.py

    Args:
        learn: Learner object that will be used for prediction
        dl: DataLoader the model will use to load samples
        with_loss: If True, it will also return the loss on each prediction
        n_batch: Number of batches to predict. If not specified, it will run the predictions for n batches
            where n = sample size // BATCH_SIZE
        pbar: ProgressBar object
    """

    # Note: In Fastai, for DatasetType.Train, only the output of complete minibatches is computed. Ie if one has 101 images,
    # and uses a minibatch size of 16, then len(feats) is 96 and not 101. For DatasetType.Valid this is not the case,
    # and len(feats) is as expected 101. A way around this is to use DatasetType.Fix instead when referring to the training set.
    # See e.g. issue: https://forums.fast.ai/t/get-preds-returning-less-results-than-length-of-original-dataset/34148
    if dl == DatasetType.Train:
        dl = DatasetType.Fix

    lf = learn.loss_func if with_loss else None
    return fastai.basic_train.get_preds(
        learn.model,
        dl,
        cb_handler=CallbackHandler(learn.callbacks),
        activ=_loss_func2activ(learn.loss_func),
        loss_func=lf,
        n_batch=n_batch,
        pbar=pbar,
    )


class TrainMetricsRecorder(LearnerCallback):
    _order = -20  # Needs to run before the recorder

    def __init__(
        self, learn: Learner, n_batch: int = None, show_graph: bool = False
    ):
        """Fastai Train hook to evaluate metrics on train and validation set for every epoch.

        This class works with the metrics functions whose signature is fn(input:Tensor, targs:Tensor),
        e.g. fastai.metrics.accuracy and error_rate.
        For custom metrics, see https://docs.fast.ai/metrics.html#Creating-your-own-metric

        Note, Learner's Recorder callback tracks the metrics and loss on the validation set and
        ShowGraph callback plots the loss on train and validation sets while training.
        TrainMetricsRecorder, on the other hand, records the metrics on the training set and plot them as well.

        Arguments:
            n_batch (int): Number of train batches to use when evaluate metrics on the training set.
                If None, use all the training set which will take longer time.
            show_graph (bool): If True, draw metrics after each epoch. If multiple metrics have set,
                it draws only the first metrics graph.

        Examples:
            >>> learn = cnn_learner(data, model, metrics=[accuracy])
            >>> train_metrics_cb = TrainMetricsRecorder(learn, n_batch=1)
            >>> learn.callbacks.append(train_metrics_cb)
            >>> learn.fit(epochs=10, lr=0.001)
            >>> train_metrics_cb.plot()

            or

            >>> learn = cnn_learner(data, model, metrics=[accuracy, error_rate],
            ...     callback_fns=[partial(
            ...         TrainMetricsRecorder,
            ...         n_batch=len(data.valid_ds)//BATCH_SIZE,
            ...         show_graph=True
            ...     )]
            ... )
    )])

        """
        super().__init__(learn)

        # Check number of batches we will evaluate on with the metrics.
        if n_batch:
            assert n_batch > 0

        self.n_batch = n_batch
        self.show_graph = show_graph

    def on_train_begin(
        self, pbar: PBar, metrics: List, n_epochs: int, **kwargs: Any
    ):
        self.has_metrics = metrics and len(metrics) > 0
        self.has_val = hasattr(self.learn.data, "valid_ds")

        # Result table and graph variables
        self.learn.recorder.silent = (
            True
        )  # Mute recorder. This callback will printout results instead.
        self.pbar = pbar
        self.names = ["epoch", "train_loss"]
        if self.has_val:
            self.names.append("valid_loss")
        # Add metrics names
        self.metrics_names = [m_fn.__name__ for m_fn in metrics]
        for m in self.metrics_names:
            self.names.append("train_" + m)
            if self.has_val:
                self.names.append("valid_" + m)
        self.names.append("time")
        self.pbar.write(self.names, table=True)

        self.n_epochs = n_epochs
        self.valid_metrics = []
        self.train_metrics = []

        # Reset graph
        self._fig = None
        self._axes = None
        self._display = None

    def on_epoch_begin(self, **kwargs: Any):
        self.start_epoch = time()
        self.y = []  # Target class labels from the last epoch
        self.out = []  # Outputs from the last epoch

    def on_batch_end(
        self,
        train: bool,
        num_batch: int,
        last_target: Tensor,
        last_output: Tensor,
        **kwargs: Any,
    ):
        if (
            train
            and (self.n_batch is None or self.n_batch > num_batch)
            and self.has_metrics
        ):
            self.y.append(last_target.cpu())
            self.out.append(last_output.cpu())

    def on_epoch_end(
        self,
        epoch: int,
        smooth_loss: Tensor,
        metrics: List,
        last_metrics: List,
        pbar: PBar,
        **kwargs: Any,
    ):
        stats = [epoch, smooth_loss]
        if self.has_val:
            stats.append(last_metrics[0])  # validation loss

        if self.has_metrics:
            # Evaluate metrics on the training set
            tr_lm = [
                m_fn(torch.stack(self.out), torch.stack(self.y))
                for m_fn in metrics
            ]
            self.train_metrics.append(tr_lm)

            # Get evaluation metrics on the validation set (computed by learner)
            if self.has_val:
                vl_lm = last_metrics[1:]
                self.valid_metrics.append(vl_lm)

            # Prepare result table values
            for i in range(len(metrics)):
                stats.append(tr_lm[i])
                if self.has_val:
                    stats.append(vl_lm[i])

        # Write to result table
        self._format_stats(stats)

        # Plot (update) metrics for every end of epoch
        if self.show_graph and len(self.train_metrics) > 0:
            self._plot(True)

    def _format_stats(self, stats: TensorOrNumList) -> None:
        """Format stats before printing. Note, this does the same thing as Recorder's"""
        str_stats = []
        for name, stat in zip(self.names, stats):
            str_stats.append(
                "#na#"
                if stat is None
                else str(stat)
                if isinstance(stat, int)
                else f"{stat:.6f}"
            )
        str_stats.append(format_time(time() - self.start_epoch))
        self.pbar.write(str_stats, table=True)

    def _plot(self, update=False):
        if not self._fig:
            # init graph
            self._fig, self._axes = plt.subplots(
                len(self.train_metrics[0]),
                1,
                figsize=(6, 4 * len(self.train_metrics[0])),
            )
            self._axes = (
                self._axes.flatten()
                if len(self.train_metrics[0]) > 1
                else [self._axes]
            )
            plt.close(self._fig)

        # Plot each metrics as a subplot
        for i, ax in enumerate(self._axes):
            ax.clear()

            # Plot training set results
            tr_m = [met[i] for met in self.train_metrics]
            x_axis = [i for i in range(len(tr_m))]
            ax.plot(x_axis, tr_m, label="Train")

            # Plot validation set results
            maybe_y_bounds = [
                -0.05,
                1.05,
                min(Tensor(tr_m)),
                max(Tensor(tr_m)),
            ]
            if len(self.valid_metrics) > 0:
                vl_m = [met[i] for met in self.valid_metrics]
                ax.plot(x_axis, vl_m, label="Validation")
                maybe_y_bounds.extend([min(Tensor(vl_m)), max(Tensor(vl_m))])

            x_bounds = (-0.05, self.n_epochs - 0.95)
            y_bounds = (min(maybe_y_bounds) - 0.05, max(maybe_y_bounds) + 0.05)
            ax.set_xlim(x_bounds)
            ax.set_ylim(y_bounds)

            ax.set_ylabel(self.metrics_names[i])
            ax.set_xlabel("Epochs")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="upper right")

        if update:
            if not self._display:
                self._display = display(self._fig, display_id=True)
            else:
                self._display.update(self._fig)

    def plot(self):
        """Plot metrics graph"""
        if len(self.train_metrics) == 0:
            raise ValueError("No records to plot.")

        # Number of metrics on training set and validation set should be the same
        if len(self.valid_metrics) > 0:
            assert len(self.train_metrics[0]) == len(self.valid_metrics[0])

        self._plot()
        display(self._fig)

    def last_train_metrics(self):
        """Train set metrics from the last epoch"""
        return self.train_metrics[-1]

    def last_valid_metrics(self):
        """Validation set metrics from the last epoch"""
        return self.valid_metrics[-1]
