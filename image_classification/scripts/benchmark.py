import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import pandas as pd
import time
import torch
import shutil

from utils_ic.datasets import unzip_urls
from argparse import RawTextHelpFormatter, Namespace
from enum import Enum
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from functools import partial
from typing import Callable, Union, List, Any
from pathlib import Path

Time = float

# ================================================ #
# Default parameters                               #
# ================================================ #

# DATA and OUTPUT
DATA_DIR = "benchmark_data_dir"
RESULTS_FILE = "results.csv"

# NUMBER OF TIMES TO RUN ALL PERMUTATIONS
REPEAT = 1

# ENABLE EARLY STOPPING
EARLY_STOPPING = True

# DEFAULT HYPERPARAMS TO EXPLORE:
LRS = [1e-4]
EPOCHS = [5]
BATCH_SIZES = [16]
IM_SIZES = [299]
ARCHITECTURES = ["resnet18"]
TRANSFORMS = [True]
DROPOUTS = [0.5]
WEIGHT_DECAYS = [0.01]
TRAINING_SCHEDULES = ["head_first_then_body"]
DISCRIMINATIVE_LRS = [False]
ONE_CYCLE_POLICIES = [False]

# TODO add precision (fp16, fp32), add momentum

# ================================================ #
# Messages                                         #
# ================================================ #

argparse_desc_msg = (
    lambda: f"""
This script is used to benchmark the different hyperparameters when it comes to doing image classification.

This script will run all permutations of the parameters that are passed in.

This script will run these tests on datasets provided in this repo. It will
create a temporary data directory, and delete it at the end.

This script uses accuracy as the evaluation metric.

Use [-W ignore] to ignore warning messages when running the script.
"""
)

argparse_epilog_msg = (
    lambda: f"""
Default parameters are:

LR:                {LRS[0]}
EPOCH:             {EPOCHS[0]}
IM_SIZE:           {IM_SIZES[0]}
BATCH_SIZE:        {BATCH_SIZES[0]}
ARCHITECTURE:      {ARCHITECTURES[0]}
TRANSFORMS:        {TRANSFORMS[0]}
DROPOUT:           {DROPOUTS[0]}
WEIGHT_DECAY:      {WEIGHT_DECAYS[0]}
TRAINING_SCHEDULE: {TRAINING_SCHEDULES[0]}
DISCRIMINATIVE_LR: {DISCRIMINATIVE_LRS[0]}
ONE_CYCLE_POLICY:  {ONE_CYCLE_POLICIES[0]}

Example usage:

# Test the effect of 3 learning rates on 3 batch sizes
$ python benchmark.py -l 1e-3 1e-4 1e-5 -bs 8 16 32 -o learning_rate_batch_size.csv

# Test the effect of one cycle policy without using discriminative learning rates over 5 runs
$ python benchmark.py -dl False -ocp True False -r 5 -o ocp_dl.csv

# Test different architectures and image sizes
$ python benchmark.py -a squeezenet1_1 resenet18 resnet50 -is 299 499 -o arch_im_sizes.csv

# Test different training schedules over 3 runs
$ python benchmark.py -ts body_only head_first_then_body -r 3 -o training_schedule.csv

---

To view results, we recommend using pandas dataframes:

```
import pandas as pd
df = pd.read_csv("results.csv", index_col=[0, 1, 2])
```

"""
)

param_msg = (
    lambda: f"""
--------------------------------------------------------------------
RUN # {i+1} of {len(permutations)} | REPEAT # {r+1} of {args.repeat}
--------------------------------------------------------------------
LR:                {LR}
EPOCH:             {EPOCH}
IM_SIZE:           {IM_SIZE}
BATCH_SIZE:        {BATCH_SIZE}
ARCHITECTURE:      {ARCHITECTURE}
TRANSFORMS:        {TRANSFORM}
DROPOUT:           {DROPOUT}
WEIGHT_DECAY:      {WEIGHT_DECAY}
TRAINING_SCHEDULE: {TRAINING_SCHEDULE}
DISCRIMINATIVE_LR: {DISCRIMINATIVE_LR}
ONE_CYCLE_POLICY:  {ONE_CYCLE_POLICY}
"""
)

result_msg = (
    lambda: f"""
{os.path.basename(path)}:
    duration: {val['duration']}
    accuracy: {val['accuracy']}
"""
)

time_msg = (
    lambda: f"""
===================================================================
Total Time elapsed: {end - start} seconds
===================================================================
"""
)
# ================================================ #


class TrainingSchedule(Enum):
    head_only = 0
    body_only = 1
    head_first_then_body = 2


class Architecture(Enum):
    resnet18 = partial(models.resnet18)
    resnet34 = partial(models.resnet34)
    resnet50 = partial(models.resnet50)
    squeezenet1_1 = partial(models.squeezenet1_1)


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _get_parser() -> Namespace:

    parser = argparse.ArgumentParser(
        description=argparse_desc_msg(),
        epilog=argparse_epilog_msg(),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--lr",
        "-l",
        dest="lrs",
        nargs="+",
        help="learning rate - recommended options: [1e-3, 1e-4, 1e-5] ",
        default=LRS,
        type=float,
    )
    parser.add_argument(
        "--epoch",
        "-e",
        dest="epochs",
        nargs="+",
        help="epochs - recommended options: [3, 5, 10, 15]",
        default=EPOCHS,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        dest="batch_sizes",
        nargs="+",
        help="batch sizes - recommended options: [8, 16, 32, 64]",
        default=BATCH_SIZES,
        type=int,
    )
    parser.add_argument(
        "--im-size",
        "-is",
        dest="im_sizes",
        nargs="+",
        help="image sizes - recommended options: [299, 499]",
        default=IM_SIZES,
        type=int,
    )
    parser.add_argument(
        "--architecture",
        "-a",
        dest="architectures",
        nargs="+",
        help="architecture - options: ['squeezenet1_1', 'resnet18', 'resnet34', 'resnet50']",
        default=ARCHITECTURES,
        type=str,
    )
    parser.add_argument(
        "--transform",
        "-t",
        dest="transforms",
        nargs="+",
        help="tranform - options: [True, False]",
        default=TRANSFORMS,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--dropout",
        "-d",
        dest="dropouts",
        nargs="+",
        help="dropout - recommended options: [0.5]",
        default=DROPOUTS,
        type=float,
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        dest="weight_decays",
        nargs="+",
        help="weight decay - recommended options: [0.01]",
        default=WEIGHT_DECAYS,
        type=float,
    )
    parser.add_argument(
        "--training-schedule",
        "-ts",
        dest="training_schedules",
        nargs="+",
        help="training schedule - options: ['head_only', 'body_only', 'head_first_then_body']",
        default=TRAINING_SCHEDULES,
        type=str,
    )
    parser.add_argument(
        "--discriminative-lr",
        "-dl",
        dest="discriminative_lrs",
        nargs="+",
        help="discriminative lr - options: [True, False]. To use discriminative learning rates, training schedule must not be 'head_only'",
        default=DISCRIMINATIVE_LRS,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--one-cycle-policy",
        "-ocp",
        dest="one_cycle_policies",
        nargs="+",
        help="one cycle policy - options: [True, False]",
        default=ONE_CYCLE_POLICIES,
        type=_str_to_bool,
    )
    es_parser = parser.add_mutually_exclusive_group(required=False)
    es_parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="stop training early if possible",
    )
    es_parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="do not stop training early if possible",
    )
    parser.set_defaults(early_stopping=EARLY_STOPPING)
    parser.add_argument(
        "--repeat",
        "-r",
        dest="repeat",
        help="the number of times to repeat each permutation",
        default=REPEAT,
        type=int,
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        help="the name of the output file",
        default=RESULTS_FILE,
    )
    args = parser.parse_args()

    # check all input values are valid
    for a in args.architectures:
        assert a in ["resnet18", "resnet34", "resnet50", "squeezenet1_1"]
    for a in args.training_schedules:
        assert a in ["head_only", "body_only", "head_first_then_body"]
    for a in args.transforms:
        assert a in [True, False]
    for a in args.discriminative_lrs:
        assert a in [True, False]
    for a in args.one_cycle_policies:
        assert a in [True, False]

    # if discriminative lr is on, we cannot have a 'head_only'
    # training_schedule
    if True in args.discriminative_lrs:
        assert "head_only" not in args.training_schedules

    # get mapping of architecture enum: ex. "resnet34" -->
    # Architecture.resnet34 -> models.resnet34
    architecture_params = []
    for a in args.architectures:
        architecture_params.append(Architecture[a])
    args.architectures = architecture_params

    # get mapping of training enum: ex. "head_only" -->
    # TrainingSchedule.head_only --> 0
    training_schedule_params = []
    for t in args.training_schedules:
        training_schedule_params.append(TrainingSchedule[t])
    args.training_schedules = training_schedule_params

    return args


def _get_permutations(args: Namespace) -> List[Tuple[Any]]:
    """
    Returns a list of all permutations
    """
    params = [
        args.lrs,
        args.epochs,
        args.batch_sizes,
        args.im_sizes,
        args.architectures,
        args.transforms,
        args.dropouts,
        args.weight_decays,
        args.training_schedules,
        args.discriminative_lrs,
        args.one_cycle_policies,
    ]
    permutations = list(itertools.product(*params))
    return permutations


def _get_data_bunch(
    path: Union[Path, str], transform: bool, im_size: int, bs: int
) -> ImageDataBunch:
    """
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
    data = _get_data_bunch(data_path, TRANSFORM, IM_SIZE, BATCH_SIZE)

    # callbacks to pass to learner
    callbacks = list()
    if stop_early:
        callbacks.append(
            partial(
                EarlyStoppingCallback,
                monitor="accuracy",
                min_delta=0.01,  # conservative
                patience=3,
            )
        )

    # create learner
    learn = cnn_learner(
        data, arch, metrics=accuracy, ps=p, callback_fns=callbacks
    )

    # discriminative learning
    original_lr = lr
    lr = slice(lr, 3e-3) if discriminative_lr else lr

    # create fitter function
    def fitter(
        one_cycle_policy: bool = one_cycle_policy,
        epoch: int = epoch,
        lr: Union[slice, int] = lr,
        wd: int = wd,
    ) -> Callable:
        if one_cycle_policy:
            return partial(learn.fit_one_cycle, cyc_len=epoch, lr=lr, wd=wd)
        else:
            return partial(learn.fit, epoch=epoch, lr=lr, wd=wd)

    # one cycle policy & wd
    fit = partial(fitter, one_cycle_policy=one_cycle_policy, wd=wd)

    # training schedule
    if train_schedule is TrainingSchedule.head_only:
        if discriminative_lr:
            raise Exception(
                "Cannot run discriminative_lr if training schedule is head_only."
            )
        else:
            fit(epoch=epoch, lr=lr)

    elif train_schedule is TrainingSchedule.body_only:
        learn.unfreeze()
        fit(epoch=epoch, lr=lr)

    elif train_schedule is TrainingSchedule.head_first_then_body:
        head = epoch / 4
        fit(epoch=head, lr=original_lr)  # train head on 25% of epochs
        learn.unfreeze()
        fit(epoch=epoch - head, lr=lr)  # train body on remaining 75% of epochs

    end = time.time()
    duration = end - start

    return learn, duration


def _serialize_permutations(p: Tuple[Any]) -> str:
    p = iter(p)
    return (
        f"LR: {next(p)}, EPOCHS: {next(p)}, BATCH_SIZE: {next(p)},"
        f"IM_SIZE: {next(p)}, ARCHITECTURE: {next(p)}, TRANSFORM: {next(p)},"
        f"DROPOUT: {next(p)}, WEIGHT_DECAY: {next(p)}"
        f"TRAINING_SCHEDULE: {next(p)}, DISCRIMINATIVE_LR: {next(p)},"
        f"ONE_CYCLE_POLICY: {next(p)}"
    )


if __name__ == "__main__":

    start = time.time()
    args = _get_parser()
    datasets = unzip_urls(DATA_DIR)
    permutations = _get_permutations(args)

    results = dict()
    for r in range(args.repeat):

        results[r] = dict()
        for i, p in enumerate(permutations):
            print(
                f"Running {i+1} of {len(permutations)} permutations. Repeat {r+1} of {args.repeat}."
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
                    args.early_stopping,
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
    results_df.to_csv(args.output)
    shutil.rmtree(DATA_DIR)

    end = time.time()
    print(time_msg())
