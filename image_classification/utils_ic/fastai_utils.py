# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import matplotlib.pyplot as plt
from typing import List, Any
from fastai.callback import Callback
from fastai.core import PBar
from torch import Tensor
from matplotlib.ticker import MaxNLocator


class TrainMetricsRecorder(Callback):
    def __init__(self, n_batch:int=None, show_graph:bool=False):
        """Fastai Train hook. Calculate metrics on train and validation set for every epoch.

        Works with the metrics functions whose signature is fn(input:Tensor, targs:Tensor),
        e.g. fastai.metrics.accuracy and error_rate.

        Arguments:
            n_batch (int): Number of train batches to use when evaluate metrics on the training set.
                If None, use all the training set which will take longer time.
            show_graph (bool): If True, draw metrics after each epoch. If multiple metrics have set,
                it draws only the first metrics graph.

        Examples:
            >>> learn = cnn_learner(data, models.resnet50, metrics=[accuracy])
            >>> train_metrics_cb = TrainMetricsRecorder(n_batch=1, show_graph=True)
            >>> learn.fit(epochs=10, lr=0.001, callbacks=[train_metrics_cb])
            >>> train_metrics_cb.plot()
        """
        self.n_batch = n_batch
        self.show_graph = show_graph

        self.epochs = None  # x-axis, i.e. [0, 1, ... n_epochs-1]
        self.metric_names = None
        self.valid_metrics = None
        self.train_metrics = None
        self.y = None       # Target class labels from the last epoch
        self.out = None     # Outputs from the last epoch

    def on_train_begin(self, metrics:List, n_epochs:int, **kwargs:Any):
        if not metrics or len(metrics) == 0:
            import warnings
            warnings.warn(f"{self.__class__.__name__}: No metrics to record. Metrics should be set to the learner.")
        else:
            self.epochs = [i for i in range(0, n_epochs)]  # Fastai's epoch number starts from 0
            self.metric_names = [m_fn.__name__ for m_fn in metrics]
            self.valid_metrics = []
            self.train_metrics = []

    def on_epoch_begin(self, **kwargs:Any):
        self.y = []
        self.out = []

    def on_loss_begin(self, train:bool, num_batch:int, metrics:List, last_target:Tensor, last_output:Tensor,
                      **kwargs:Any):
        """Callback on loss begin. This is being called between model prediction and backpropagation while training."""
        if train and (self.n_batch is None or self.n_batch > num_batch) and (metrics and len(metrics) > 0):
            self.y.append(last_target.cpu())
            self.out.append(last_output.cpu())

    def on_epoch_end(self, epoch:int, metrics:List, last_metrics:List, pbar:PBar, **kwargs:Any):
        if metrics and len(metrics) > 0:
            # Metrics on the training set.
            train_metrics = []
            for m_fn in metrics:
                train_metrics.append(m_fn(torch.stack(self.out), torch.stack(self.y)))
            self.train_metrics.append(train_metrics)

            # Metrics on the validation set. Note, last_metrics[0] is the validation loss
            if last_metrics and len(last_metrics) > 1:
                self.valid_metrics.append(last_metrics[1:])

            # Plot 1st metrics for every end of epoch
            if self.show_graph:
                self._update_pbar_graph(pbar)

            # Follow fastai's callback ShowGraph's return value
            return {}

    def _update_pbar_graph(self, pbar:PBar):
        """Update Learner.Recorder.pbar graphs. Only the 1st metrics will be plotted"""
        train_metrics_0 = [met[0] for met in self.train_metrics]
        valid_metrics_0 = []

        # Add 1st metrics on the training set to the graph
        x_axis = [i for i in range(0, len(train_metrics_0))]
        pbar.names = ["Train"]
        graphs = [(x_axis, train_metrics_0)]

        # Add 1st metrics on the validation set to the graph if exists
        if len(self.valid_metrics) > 0:
            valid_metrics_0 = [met[0] for met in self.valid_metrics]
            pbar.names.append("Validation")
            graphs.append((x_axis, valid_metrics_0))

        # +- 0.05 boundary padding
        x_bounds = (-0.05, len(self.epochs) - 0.95)
        y_bounds = (
            min(-0.05, min(Tensor(train_metrics_0)), min(Tensor(valid_metrics_0))) - 0.05,
            max(1.05, max(Tensor(train_metrics_0)), max(Tensor(valid_metrics_0))) + 0.05
        )
        pbar.update_graph(graphs, x_bounds, y_bounds)

        try:
            # Draw x and y axes names
            pbar.ax.set_ylabel(self.metric_names[0])
            pbar.ax.set_xlabel("Epochs")
            pbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            pbar.out2.update(pbar.ax.figure)
        except AttributeError:
            # ax and out2 attributes may not be set yet
            pass

    def plot(self):
        """Plot metrics graph"""
        if self.train_metrics is None:
            raise ValueError("No records to plot.")

        # Number of metrics on training set and validation set should be the same
        if len(self.valid_metrics) > 0:
            assert len(self.train_metrics[0]) == len(self.valid_metrics[0])

        fig, axes = plt.subplots(len(self.train_metrics[0]), 1, figsize=(6, 4 * len(self.train_metrics[0])))
        axes = axes.flatten() if len(self.train_metrics[0]) != 1 else [axes]
        # Plot each metrics as a subplot
        for i, ax in enumerate(axes):
            train_metrics = [met[i] for met in self.train_metrics]
            valid_metrics = []
            ax.plot(self.epochs, train_metrics, label="Train")

            if len(self.valid_metrics) > 0:
                valid_metrics = [met[i] for met in self.valid_metrics]
                ax.plot(self.epochs, valid_metrics, label="Validation")

            x_bounds = (-0.05, len(self.epochs) - 0.95)
            y_bounds = (
                min(-0.05, min(Tensor(train_metrics)), min(Tensor(valid_metrics))) - 0.05,
                max(1.05, max(Tensor(train_metrics)), max(Tensor(valid_metrics))) + 0.05
            )
            ax.set_xlim(x_bounds)
            ax.set_ylim(y_bounds)

            ax.set_ylabel(self.metric_names[i])
            ax.set_xlabel("Epochs")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()

        # To use this in python script, may use >>> if not IN_NOTEBOOK: plot_sixel(fig)
