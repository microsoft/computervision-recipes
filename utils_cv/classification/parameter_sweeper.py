# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
from enum import Enum
from functools import partial
import itertools
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple, Union

from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from fastai.vision import (
    cnn_learner,
    get_transforms,
    ImageDataBunch,
    ImageList,
    imagenet_stats,
    Learner,
    models,
)
from matplotlib.axes import Axes
from matplotlib.text import Annotation
import pandas as pd


Time = float
parameter_flag = "PARAMETERS"


class TrainingSchedule(Enum):
    head_only = ("head_only",)
    body_only = ("body_only",)
    head_first_then_body = "head_first_then_body"


class Architecture(Enum):
    resnet18 = partial(models.resnet18)
    resnet34 = partial(models.resnet34)
    resnet50 = partial(models.resnet50)
    squeezenet1_1 = partial(models.squeezenet1_1)


class DataFrameAlreadyCleaned(Exception):
    pass


def clean_sweeper_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleans up dataframe outputed from sweeper

    Cleans up experiment paramter strings in {df} by removing all experiment
    parameters that held constant through each experiment. This method uses a
    variable <parameter_flag> to search for strings.

    Args:
        df (pd.DataFrame): dataframe to clean up
    Raises:
        DataFrameAlreadyCleaned
    Return:
        pd.DataFrame: df with renamed experiment parameter strings
    """
    text = df.to_html()

    if parameter_flag not in text:
        raise DataFrameAlreadyCleaned

    text = re.findall(fr">\s{{0,1}}{parameter_flag}\s{{0,1}}(.*?)</th>", text)

    sets = [set(t.split("|")) for t in text]
    intersection = sets[0].intersection(*sets)

    html = df.to_html()
    for i in intersection:
        html = html.replace(i, "")
    html = html.replace("PARAMETERS", "P:")
    html = html.replace("|", " ")

    return pd.read_html(html, index_col=[0, 1, 2])[0]


def add_value_labels(
    ax: Axes, spacing: int = 5, percentage: bool = False
) -> None:
    """ Add labels to the end of each bar in a bar chart.

    Overwrite labels on axes if they already exist.

    Args:
        ax (Axes): The matplotlib object containing the axes of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        percentage (bool): if y-value is a percentage
    """
    for child in ax.get_children():
        if isinstance(child, Annotation):
            child.remove()

    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = (
            "{:.2f}%".format(y_value * 100)
            if percentage
            else "{:.1f}".format(y_value)
        )

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va="bottom",  # Vertically align label
        )


def plot_sweeper_df(
    df: pd.DataFrame,
    sort_by: str = None,
    figsize: Tuple[int, int] = (12, 8),
    show_cols: List[str] = None,
) -> None:
    """ Visualize df outputed from sweeper

    Visualize graph from {df}, which should contain columns "accuracy" and
    "duration". Columns not titled "accuracy" or "duration" will also be
    rendered.

    Args:
        df (pd.DataFrame): the dataframe to visualize.
        sort_by (str): whether to sort visualization by accuracy or duration.
        figsize (Tuple[int, int]): as defined in matplotlib.
        show_cols (List[str]): a list of columns in the df to show
    Raises:
        ValueError: if {sort_by} is an invalid value, if elements of
        {show_cols} is not a valid column name, or if {sort_by} is not in
        {show_cols} if it is used.
    """
    cols = list(df.columns.values) if show_cols is None else show_cols

    if not set(cols) <= set(list(df.columns.values)):
        raise ValueError("values of {show_cols} is not found {df}.")

    if sort_by is not None and sort_by not in cols:
        raise ValueError(
            "{sort_by} must be in {show_cols} if {show_cols} is used."
        )

    if sort_by:
        df = df.sort_values(by=sort_by)

    axes = df[cols].plot.bar(
        rot=90, subplots=True, legend=False, figsize=figsize
    )

    assert len(cols) == len(axes)

    for col, ax in zip(cols, axes):
        top_val = df[col].max()
        ax.set_ylim(top=top_val * 1.2)
        add_value_labels(ax)

        if col in ["accuracy"]:
            add_value_labels(ax, percentage=True)
            ax.set_title("Accuracy (%)")
            ax.set_ylabel("%")
        if col in ["duration"]:
            ax.set_title("Training Duration (seconds)")
            ax.set_ylabel("seconds")


class ParameterSweeper:
    """ Test different permutations of a set of parameters.

    Attributes:
        param_order <Tuple[str]>: A fixed ordering of parameters (to match the ordering of <params>)
        default_params <Dict[str, Any]>: A dict of default parameters
        params <Dict[str, List[Any]]>: The parameters to run experiments on
    """

    default_params = dict(
        learning_rate=1e-4,
        epoch=15,
        batch_size=16,
        im_size=299,
        architecture=Architecture.resnet18,
        transform=True,
        dropout=0.5,
        weight_decay=0.01,
        training_schedule=TrainingSchedule.head_first_then_body,
        discriminative_lr=False,
        one_cycle_policy=True,
        metric_name="accuracy",
    )

    def __init__(self, **kwargs) -> None:
        """
        Initialize class with default params if kwargs is empty.
        Otherwise, initialize params with kwargs.
        """
        self.params = OrderedDict(
            learning_rate=[self.default_params.get("learning_rate")],
            epochs=[self.default_params.get("epoch")],
            batch_size=[self.default_params.get("batch_size")],
            im_size=[self.default_params.get("im_size")],
            architecture=[self.default_params.get("architecture")],
            transform=[self.default_params.get("transform")],
            dropout=[self.default_params.get("dropout")],
            weight_decay=[self.default_params.get("weight_decay")],
            training_schedule=[self.default_params.get("training_schedule")],
            discriminative_lr=[self.default_params.get("discriminative_lr")],
            one_cycle_policy=[self.default_params.get("one_cycle_policy")],
            metric_name=[self.default_params.get("metric_name")],
        )

        self.param_order = tuple(self.params.keys())
        self.update_parameters(**kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        """ Returns parameters to test on if run() is called. """
        return self.params

    @property
    def permutations(self) -> List[Tuple[Any]]:
        """ Returns a list of all permutations, expressed in tuples. """
        params = tuple([self.params[k] for k in self.param_order])
        permutations = list(itertools.product(*params))
        return permutations

    @staticmethod
    def _get_data_bunch(
        path: Union[Path, str], transform: bool, im_size: int, bs: int
    ) -> ImageDataBunch:
        """
        Create ImageDataBunch and return it. TODO in future version is to allow
        users to pass in their own image bunch or their own Transformation
        objects (instead of using fastai's <get_transforms>)

        Args:
            path (Union[Path, str]): path to data to create databunch with
            transform (bool): a flag to set fastai default transformations (get_transforms())
            im_size (int): image size of databunch
            bs (int): batch size of databunch
        Returns:
            ImageDataBunch
        """
        path = path if type(path) is Path else Path(path)
        tfms = get_transforms() if transform else None
        return (
            ImageList.from_folder(path)
            .split_by_rand_pct(valid_pct=0.33)
            .label_from_folder()
            .transform(tfms=tfms, size=im_size)
            .databunch(bs=bs)
            .normalize(imagenet_stats)
        )

    @staticmethod
    def _early_stopping_callback(
        metric: str = "accuracy", min_delta: float = 0.01, patience: int = 3
    ) -> partial:
        """ Returns an early stopping callback. """
        return partial(
            EarlyStoppingCallback,
            monitor=metric,
            min_delta=min_delta,  # conservative
            patience=patience,
        )

    @staticmethod
    def _serialize_permutations(p: Tuple[Any]) -> str:
        """ Serializes all parameters as a string that uses {parameter_flag}. """
        p = iter(p)
        return (
            f"{parameter_flag} "
            f"[learning_rate: {next(p)}]|[epochs: {next(p)}]|[batch_size: {next(p)}]|"
            f"[im_size: {next(p)}]|[arch: {next(p).name}]|"
            f"[transforms: {next(p)}]|[dropout: {next(p)}]|"
            f"[weight_decay: {next(p)}]|[training_schedule: {next(p).name}]|"
            f"[discriminative_lr: {next(p)}]|[one_cycle_policy: {next(p)}]"
        )

    @staticmethod
    def _make_df_from_dict(
        results: Dict[Any, Dict[Any, Dict[Any, Dict[Any, Any]]]]
    ) -> pd.DataFrame:
        """ Converts a 4-times-nested dictionary into a multi-index dataframe. """
        return pd.DataFrame.from_dict(
            {
                (i, j, k): results[i][j][k]
                for i in results.keys()
                for j in results[i].keys()
                for k in results[i][j].keys()
            },
            orient="index",
        )

    def _param_tuple_to_dict(self, params: Tuple[Any]) -> Dict[str, Any]:
        """ Converts a tuple of parameters to a Dict. """
        return dict(
            learning_rate=params[self.param_order.index("learning_rate")],
            batch_size=params[self.param_order.index("batch_size")],
            transform=params[self.param_order.index("transform")],
            im_size=params[self.param_order.index("im_size")],
            epochs=params[self.param_order.index("epochs")],
            architecture=params[self.param_order.index("architecture")],
            dropout=params[self.param_order.index("dropout")],
            weight_decay=params[self.param_order.index("weight_decay")],
            discriminative_lr=params[
                self.param_order.index("discriminative_lr")
            ],
            training_schedule=params[
                self.param_order.index("training_schedule")
            ],
            one_cycle_policy=params[
                self.param_order.index("one_cycle_policy")
            ],
        )

    def _learn(
        self, data_path: Path, params: Tuple[Any], stop_early: bool
    ) -> Tuple[Learner, Time]:
        """
        Given a set of permutations, create a learner to train and validate on
        the dataset.
        Args:
            data_path (Path): The location of the data to use
            params (Tuple[Any]): The set of parameters to train and validate on
            stop_early (bool): Whether or not to stop early if the evaluation
            metric does not improve
        Returns:
            Tuple[Learner, Time]: Learn object from Fastai and the duration in
            seconds it took.
        """
        start = time.time()
        params = self._param_tuple_to_dict(params)

        transform = params["transform"]
        im_size = params["im_size"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        architecture = params["architecture"]
        dropout = params["dropout"]
        learning_rate = params["learning_rate"]
        discriminative_lr = params["discriminative_lr"]
        training_schedule = params["training_schedule"]
        one_cycle_policy = params["one_cycle_policy"]
        weight_decay = params["weight_decay"]

        data = self._get_data_bunch(data_path, transform, im_size, batch_size)

        callbacks = list()
        if stop_early:
            callbacks.append(ParameterSweeper._early_stopping_callback())

        learn = cnn_learner(
            data,
            architecture.value,
            metrics=accuracy,
            ps=dropout,
            callback_fns=callbacks,
        )

        head_learning_rate = learning_rate
        body_learning_rate = (
            slice(learning_rate, 3e-3) if discriminative_lr else learning_rate
        )

        def fit(
            learn: Learner, e: int, lr: Union[slice, float], wd=float
        ) -> partial:
            """ Returns a partial func for either fit_one_cycle or fit
            depending on <one_cycle_policy> """
            return (
                partial(learn.fit_one_cycle, cyc_len=e, max_lr=lr, wd=wd)
                if one_cycle_policy
                else partial(learn.fit, epochs=e, lr=lr, wd=wd)
            )

        if training_schedule is TrainingSchedule.head_only:
            if discriminative_lr:
                raise Exception(
                    "Cannot run discriminative_lr if training schedule is head_only."
                )
            else:
                fit(learn, epochs, body_learning_rate, weight_decay)()

        elif training_schedule is TrainingSchedule.body_only:
            learn.unfreeze()
            fit(learn, epochs, body_learning_rate, weight_decay)()

        elif training_schedule is TrainingSchedule.head_first_then_body:
            head_epochs = epochs // 4
            fit(learn, head_epochs, head_learning_rate, weight_decay)()
            learn.unfreeze()
            fit(
                learn, epochs - head_epochs, body_learning_rate, weight_decay
            )()

        end = time.time()
        duration = end - start

        return learn, duration

    def update_parameters(self, **kwargs) -> "ParameterSweeper":
        """ Update the class object's parameters.
        If kwarg key is not in an existing param key, then raise exception.
        If the kwarg value is None, pass.
        Otherwise overwrite the corresponding self.params key.
        """
        for k, v in kwargs.items():
            if k not in self.params.keys():
                raise Exception("Parameter {k} is invalid.")
            if v is None:
                continue
            self.params[k] = v

        return self

    def run(
        self,
        datasets: List[Path],
        reps: int = 3,
        early_stopping: bool = False,
        metric_fct=None,
    ) -> pd.DataFrame:
        """ Performs the experiment.
        Iterates through the number of specified <reps>, the list permutations
        as defined in this class, and the <datasets> to calculate evaluation
        metrics and duration for each run.

        WARNING: this method can take a long time depending on your experiment
        definition.

        Args:
            datasets (List[Path]): A list of datasets to iterate over.
            reps (int): The number of runs to loop over.
            early_stopping (bool): Whether we want to perform early stopping.
        Returns:
            pd.DataFrame: a multi-index dataframe with the results stored in it.
        """

        res = dict()
        for rep in range(reps):
            print

            res[rep] = dict()
            for i, permutation in enumerate(self.permutations):
                print(
                    f"Running {i+1} of {len(self.permutations)} permutations. "
                    f"Repeat {rep+1} of {reps}."
                )

                stringified_permutation = self._serialize_permutations(
                    permutation
                )
                res[rep][stringified_permutation] = dict()
                for dataset in datasets:

                    data_name = os.path.basename(dataset)

                    res[rep][stringified_permutation][data_name] = dict()

                    learn, duration = self._learn(
                        dataset, permutation, early_stopping
                    )

                    if metric_fct is None:
                        _, metric = learn.validate(
                            learn.data.valid_dl, metrics=[accuracy]
                        )

                    else:
                        metric = metric_fct(learn)

                    res[rep][stringified_permutation][data_name][
                        "duration"
                    ] = duration
                    res[rep][stringified_permutation][data_name][
                        self.params["metric_name"]
                    ] = float(metric)

                    learn.destroy()

        return self._make_df_from_dict(res)
