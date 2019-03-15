import itertools
import pandas as pd
import re
import time
import torch

from IPython.display import display, HTML
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from functools import partial
from typing import Union, List, Any, Dict
from pathlib import Path

Time = float


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
    TODO
    Prettify df by
    removing redundant params
    """
    text = df.to_html()
    text = re.findall(r"<th>\s{0,1}PARAMETERS\s{0,1}(.*?)</th>", text)

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
    TODO
    """

    def add_value_labels(ax, spacing=5, percentage=False):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
            percentage (bool): if y-value is a percentage
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
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, spacing),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha="center",  # Horizontally center label
                va="bottom",  # Vertically align label differently
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


class Benchmark:

    learning_rate = 1e-4
    epoch = 15
    batch_size = 16
    im_size = 299
    architecture = Architecture.resnet18
    transform = True
    dropout = 0.5
    weight_decay = 0.01
    training_schedule = TrainingSchedule.head_first_then_body
    discriminative_lr = True
    one_cycle_policy = True

    def __init__(self, **kwargs) -> None:
        """
        TODO
        """
        self.param_seq = (
            "learning_rate",
            "epochs",
            "batch_size",
            "im_size",
            "architecture",
            "transform",
            "dropout",
            "weight_decay",
            "training_schedule",
            "discriminative_lr",
            "one_cycle_policy",
        )

        self.params = dict(
            learning_rate=[self.learning_rate],
            epochs=[self.epoch],
            batch_size=[self.batch_size],
            im_size=[self.im_size],
            architecture=[self.architecture],
            transform=[self.transform],
            dropout=[self.dropout],
            weight_decay=[self.weight_decay],
            training_schedule=[self.training_schedule],
            discriminative_lr=[self.discriminative_lr],
            one_cycle_policy=[self.one_cycle_policy],
        )

        self.update_parameters(**kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.params

    @property
    def permutations(self) -> List[Tuple[Any]]:
        """
        Returns a list of all permutations, expressed in tuples
        """
        params = tuple([self.params[k] for k in self.param_seq])
        permutations = list(itertools.product(*params))
        return permutations

    @staticmethod
    def _get_data_bunch(
        path: Union[Path, str], transform: bool, im_size: int, bs: int
    ) -> ImageDataBunch:
        """
        create ImageDataBunch and return it
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
        return partial(
            EarlyStoppingCallback,
            monitor="accuracy",
            min_delta=0.01,  # conservative
            patience=3,
        )

    @staticmethod
    def _serialize_permutations(p: Tuple[Any]) -> str:
        p = iter(p)
        return (
            f"PARAMETERS "
            f"[learning_rate: {next(p)}]|[epochs: {next(p)}]|[batch_size: {next(p)}]|"
            f"[im_size: {next(p)}]|[arch: {next(p).name}]|"
            f"[transforms: {next(p)}]|[dropout: {next(p)}]|"
            f"[weight_decay: {next(p)}]|[training_schedule: {next(p).name}]|"
            f"[discriminative_lr: {next(p)}]|[one_cycle_policy: {next(p)}]"
        )

    @staticmethod
    def _make_df_from_dict(results: Dict[Any, Any]) -> pd.DataFrame:
        """ """
        return pd.DataFrame.from_dict(
            {
                (i, j, k): results[i][j][k]
                for i in results.keys()
                for j in results[i].keys()
                for k in results[i][j].keys()
            },
            orient="index",
        )

    def _learn(
        self, data_path: Path, params: Tuple[Any], stop_early: bool
    ) -> Tuple[Learner, Time]:
        """
        Create databunch, create learner with params
        return metric, duration
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
        """ update parameters
        pass if kwarg key is not in an existing param key
        or if the kwarg value is None
        Otherwise overwrite the corresponding self.param key
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
        """

        res = dict()
        for r in range(reps):

            res[r] = dict()
            for i, p in enumerate(self.permutations):
                display(
                    f"Running {i+1} of {len(self.permutations)} permutations."
                    f"Repeat {r+1} of {reps}."
                )

                permutation = self._serialize_permutations(p)
                res[r][permutation] = dict()
                for d in datasets:

                    res[r][permutation][os.path.basename(d)] = dict()

                    torch.cuda.empty_cache()

                    learn, duration = self._learn(d, p, early_stopping)

                    _, metric = learn.validate(
                        learn.data.valid_dl, metrics=[accuracy]
                    )

                    data_name = os.path.basename(d)
                    res[r][permutation][data_name]["duration"] = duration
                    res[r][permutation][data_name]["accuracy"] = float(metric)

        return self._make_df_from_dict(res)
