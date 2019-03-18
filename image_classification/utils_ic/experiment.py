import itertools
import pandas as pd
import re
import time

from utils_ic.datasets import Urls, data_path, unzip_urls
from collections import OrderedDict
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from functools import partial
from matplotlib.axes import Axes
from typing import Union, List, Any, Dict
from pathlib import Path

Time = float
parameter_flag = "PARAMETERS"


TrainingSchedule = Enum(
    "TrainingSchedule", ["head_only", "body_only", "head_first_then_body"]
)


class Architecture(Enum):
    resnet18 = partial(models.resnet18)
    resnet34 = partial(models.resnet34)
    resnet50 = partial(models.resnet50)
    squeezenet1_1 = partial(models.squeezenet1_1)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up experiment paramter strings in {df} by removing all experiment
    parameters that held constant through each experiment. This method uses a
    variable <parameter_flag> to search for strings.
    Params:
        df: dataframe to clean up
    Return df with renamed experiment parameter strings
    """
    text = df.to_html()
    text = re.findall(fr">\s{{0,1}}{parameter_flag}\s{{0,1}}(.*?)</th>", text)

    sets = []
    for t in text:
        sets.append(set(t.split("|")))
    intersection = sets[0].intersection(*sets)

    html = df.to_html()
    for i in intersection:
        html = html.replace(i, "")
    html = html.replace("PARAMETERS", "")
    html = html.replace("|", " ")

    return pd.read_html(html, index_col=[0, 1, 2])[0]


def plot_df(
    df: pd.DataFrame,
    sort_by: str = "accuracy",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Visuaize graph from {df}, which must contain columns "accuracy" and
    "duration".
    Params:
        df: the dataframe to visualize
        sort_by: whether to sort visualization by accuracy or duration
        figsize: as defined in matplotlib
    """
    if sort_by not in ("accuracy", "duration"):
        raise ValueError("{sort_by} must equal 'accuracy' or 'duration'")

    def add_value_labels(
        ax: Axes, spacing: int = 5, percentage: bool = False
    ) -> None:
        """
        Add labels to the end of each bar in a bar chart.
        Params:
            ax: The matplotlib object containing the axes of the plot to annotate.
            spacing: The distance between the labels and the bars.
            percentage: if y-value is a percentage
        """
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

    top_accuracy = df["accuracy"].max()
    top_duration = df["duration"].max()
    ax1, ax2 = df.sort_values(by=sort_by).plot.bar(
        rot=90, subplots=True, legend=False, figsize=figsize
    )
    ax1.set_title("Duration (seconds)")
    ax2.set_title("Accuracy (%)")
    ax1.set_ylabel("seconds")
    ax2.set_ylabel("%")
    ax1.set_ylim(top=top_duration * 1.2)
    ax2.set_ylim(top=top_accuracy * 1.2)
    add_value_labels(ax2, percentage=True)
    add_value_labels(ax1)


class Experiment:
    """
    Attributes:
        param_seq <Tuple[str]>: A fixed ordering of parameters (to match the ordering of <params>)
        default_params <Dict[str, Any]>: A dict of defualt parameters
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
        )

        self.param_seq = tuple(self.params.keys())
        self.update_parameters(**kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        """ Returns parameters to test on if run() is called. """
        return self.params

    @property
    def permutations(self) -> List[Tuple[Any]]:
        """ Returns a list of all permutations, expressed in tuples. """
        params = tuple([self.params[k] for k in self.param_seq])
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

        Params:
            path: path to data to create databunch with
            transform: a flag to set fastai default transformations (get_transforms())
            im_size: image size of databunch
            bs: batch size of databunch
        Returns ImageDataBunch
        """
        path = path if type(path) is Path else Path(path)
        tfms = get_transforms() if transform else None
        return (
            ImageList.from_folder(path)
            .split_by_rand_pct(valid_pct=0.33, seed=10)
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
            monitor="accuracy",
            min_delta=0.01,  # conservative
            patience=3,
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

    @classmethod
    def download_benchmark_datasets(
        cls, dest: Union[Path, str] = data_path()
    ) -> List[Path]:
        """ Download benchmark datasets to {dest}. """
        benchmark_urls = [
            Urls.fridge_objects_path,
            Urls.fashion_texture_path,
            Urls.flickr_logos_32_subset_path,
            Urls.food_101_subset_path,
            Urls.lettuce_path,
            Urls.recycle_path,
        ]
        return unzip_urls(benchmark_urls, dest)

    def _learn(
        self, data_path: Path, params: Tuple[Any], stop_early: bool
    ) -> Tuple[Learner, Time]:
        """
        Given a set of permutations, create a learner to train and validate on
        the dataset.
        Params:
            data_path: The location of the data to use
            params: The set of parameters to train and validate on
            stop_early: Whether or not to stop early if the evaluation metric
            does not improve
        Returns the Learner object from Fastai and the duration in seconds it took.
        """
        start = time.time()

        transform = params[self.param_seq.index("transform")]
        im_size = params[self.param_seq.index("im_size")]
        epochs = params[self.param_seq.index("epochs")]
        batch_size = params[self.param_seq.index("batch_size")]
        arch = params[self.param_seq.index("architecture")]
        dropout = params[self.param_seq.index("dropout")]
        learning_rate = params[self.param_seq.index("learning_rate")]
        discriminative_lr = params[self.param_seq.index("discriminative_lr")]
        training_schedule = params[self.param_seq.index("training_schedule")]
        one_cycle_policy = params[self.param_seq.index("one_cycle_policy")]
        weight_decay = params[self.param_seq.index("weight_decay")]

        data = self._get_data_bunch(data_path, transform, im_size, batch_size)

        callbacks = list()
        if stop_early:
            callbacks.append(_early_stopping_callback())

        learn = cnn_learner(
            data,
            arch.value,
            metrics=accuracy,
            ps=dropout,
            callback_fns=callbacks,
        )

        original_learning_rate = learning_rate
        learning_rate = (
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
                fit(learn, epochs, learning_rate, weight_decay)()

        elif training_schedule is TrainingSchedule.body_only:
            learn.unfreeze()
            fit(learn, epochs, original_learning_rate, weight_decay)()

        elif training_schedule is TrainingSchedule.head_first_then_body:
            head = epochs // 4
            fit(learn, head, original_learning_rate, weight_decay)()
            learn.unfreeze()
            fit(learn, epochs - head, learning_rate, weight_decay)()

        end = time.time()
        duration = end - start

        return learn, duration

    def update_parameters(self, **kwargs) -> None:
        """
        Update the class object's parameters.
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

    def run(
        self, datasets: List[Path], reps: int = 3, early_stopping: bool = False
    ) -> pd.DataFrame:
        """
        Performs the experiment. Iterates through the number of specified
        <reps>, the list permutations as defined in this class, and the
        <datasets> to calculate evaluation metrics and duration for each run.

        WARNING: this method can take a long time depending on your experiment
        definition.

        Params:
            datasets: A list of datasets to iterate over.
            reps: The number of runs to loop over.
            early_stopping: Whether we want to perform early stopping.
        Returns a multi-index dataframe with the results stored in it.
        """

        res = dict()
        for r in range(reps):

            res[r] = dict()
            for i, p in enumerate(self.permutations):
                display(
                    f"Running {i+1} of {len(self.permutations)} permutations. "
                    f"Repeat {r+1} of {reps}."
                )

                permutation = self._serialize_permutations(p)
                res[r][permutation] = dict()
                for d in datasets:

                    res[r][permutation][os.path.basename(d)] = dict()

                    learn, duration = self._learn(d, p, early_stopping)

                    _, metric = learn.validate(
                        learn.data.valid_dl, metrics=[accuracy]
                    )

                    data_name = os.path.basename(d)
                    res[r][permutation][data_name]["duration"] = duration
                    res[r][permutation][data_name]["accuracy"] = float(metric)

                    learn.destroy()

        return self._make_df_from_dict(res)
