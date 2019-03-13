import itertools
import pandas as pd
import time
import torch

from IPython.display import display, HTML
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from functools import partial
from typing import Callable, Union, List, Any
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


LR = 1e-4
EPOCH = 15
BATCH_SIZE = 16
IM_SIZE = 299
ARCHITECTURE = Architecture.resnet18
TRANSFORM = True
DROPOUT = 0.5
WEIGHT_DECAY = 0.01
TRAINING_SCHEDULE = TrainingSchedule.head_first_then_body
DISCRIMINATIVE_LR = True
ONE_CYCLE_POLICY = True


def get_parameters(
    lrs: List[float] = [LR],
    epochs = [EPOCH],
    batch_sizes = [BATCH_SIZE],
    im_sizes = [IM_SIZE],
    architectures = [ARCHITECTURE],
    transforms = [TRANSFORM],
    dropouts = [DROPOUT],
    weight_decays = [WEIGHT_DECAY],
    training_schedules = [TRAINING_SCHEDULE],
    discriminative_lrs = [DISCRIMINATIVE_LR],
    one_cycle_policies = [ONE_CYCLE_POLICY],
) -> Tuple[Any]:
    return (lrs, epochs, batch_sizes, im_sizes, architectures,
         transforms, dropouts, weight_decays, training_schedules,
         discriminative_lrs, one_cycle_policies)


def prettify_df(df: pd.DataFrame) -> None:
    return display(HTML(df.to_html().replace(",", "<br>")))


def _get_permutations(params: Tuple[Any]) -> List[Tuple[Any]]:
    """
    Returns a list of all permutations
    """
    permutations = list(itertools.product(*params))
    return permutations


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


def _early_stopping_callback(
    metric: str = "accuracy",
    min_delta: float = 0.01,
    patience: int = 3
) -> partial:
    return partial(
                EarlyStoppingCallback,
                monitor="accuracy",
                min_delta=0.01,  # conservative
                patience=3,
            )


def _learn(
    data_path: Path,
    batch_size: int,
    im_size: int,
    arch: Callable,
    transform: bool,
    epoch: int,
    lr: float,
    p: float,
    wd: float,
    train_schedule: TrainingSchedule,
    discriminative_lr: bool,
    one_cycle_policy: bool,
    stop_early: bool,
) -> Tuple[Learner, Time]:
    """
    Create databunch, create learner with params
    return metric, duration
    """
    start = time.time()

    # get databunch
    data = _get_data_bunch(data_path, transform, im_size, batch_size)

    # callbacks to pass to learner
    callbacks = list()
    if stop_early:
        callbacks.append(_early_stopping_callback)

    # create learner
    learn = cnn_learner(
        data, arch, metrics=accuracy, ps=p, callback_fns=callbacks
    )

    # discriminative learning
    original_lr = lr
    lr = slice(lr, 3e-3) if discriminative_lr else lr

    # create fitter function
    def fitter(
        learn: Learner,
        one_cycle_policy: bool = one_cycle_policy,
        epochs: int = epoch,
        lr: Union[slice, int] = lr,
        wd: int = wd,
    ) -> Callable:
        if one_cycle_policy:
            return partial(learn.fit_one_cycle, cyc_len=epochs, max_lr=lr, wd=wd)
        else:
            return partial(learn.fit, epochs=epochs, lr=lr, wd=wd)

    # one cycle policy & wd
    fit = partial(fitter, learn, one_cycle_policy=one_cycle_policy, wd=wd)

    # training schedule
    if train_schedule is TrainingSchedule.head_only:
        if discriminative_lr:
            raise Exception(
                "Cannot run discriminative_lr if training schedule is head_only."
            )
        else:
            fit(epochs=epoch, lr=lr)
            # fitter(learn)(epochs=epoch, lr=lr)()

    elif train_schedule is TrainingSchedule.body_only:
        learn.unfreeze()
        fit(epochs=epoch, lr=lr)
        # fitter(learn)(epochs=epoch, lr=lr)()

    elif train_schedule is TrainingSchedule.head_first_then_body:
        head = epoch // 4
        # fitter(learn)(epochs=head, lr=original_lr)()
        fit(epochs=head, lr=original_lr) # train head on 25% of epochs
        learn.unfreeze()
        # fitter(learn)(epochs=epoch-head, lr=original_lr)()
        fit(epochs=epoch - head, lr=lr) # train body on remaining 75% of epochs

    end = time.time()
    duration = end - start

    return learn, duration


def _serialize_permutations(p: Tuple[Any]) -> str:
    p = iter(p)
    return (
        f"lr: {next(p)}, epochs: {next(p)}, batch_size: {next(p)}, "
        f"im_size: {next(p)}, arch: {next(p).name}, "
        f"transforms: {next(p)}, dropout: {next(p)}, "
        f"weight_decay: {next(p)}, training_schedule: {next(p).name}, "
        f"discriminative_lr: {next(p)}, one_cycle_policy: {next(p)}"
    )


def benchmark(
    datasets: List[Path],
    parameters: Tuple[List[Any]],
    repeat: int,
    early_stopping: bool,
) -> pd.DataFrame:

    # UPDATE

    permutations = _get_permutations(parameters)

    results = dict()
    for r in range(repeat):

        results[r] = dict()
        for i, p in enumerate(permutations):
            display(
                f"Running {i+1} of {len(permutations)} permutations. Repeat {r+1} of {repeat}."
            )

            LR, EPOCH, BATCH_SIZE, IM_SIZE, ARCHITECTURE, TRANSFORM, DROPOUT, WEIGHT_DECAY, TRAINING_SCHEDULE, DISCRIMINATIVE_LR, ONE_CYCLE_POLICY = (
                p
            )

            serialized_permutations = _serialize_permutations(p)
            results[r][serialized_permutations] = dict()
            for d in datasets:

                results[r][serialized_permutations][
                    os.path.basename(d)
                ] = dict()

                torch.cuda.empty_cache()

                learn, duration = _learn(
                    d,
                    BATCH_SIZE,
                    IM_SIZE,
                    ARCHITECTURE.value,
                    TRANSFORM,
                    EPOCH,
                    LR,
                    DROPOUT,
                    WEIGHT_DECAY,
                    TRAINING_SCHEDULE,
                    DISCRIMINATIVE_LR,
                    ONE_CYCLE_POLICY,
                    early_stopping,
                )

                _, metric = learn.validate(
                    learn.data.valid_dl, metrics=[accuracy]
                )
                results[r][serialized_permutations][os.path.basename(d)][
                    "duration"
                ] = duration
                results[r][serialized_permutations][os.path.basename(d)][
                    "accuracy"
                ] = float(metric)

    results_df = pd.DataFrame.from_dict(
        {
            (i, j, k): results[i][j][k]
            for i in results.keys()
            for j in results[i].keys()
            for k in results[i][j].keys()
        },
        orient="index",
    )
    return results_df
