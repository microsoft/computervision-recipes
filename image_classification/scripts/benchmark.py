import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import itertools
import argparse
import torch
import shutil

from utils_ic.datasets import unzip_url, Urls
from argparse import RawTextHelpFormatter
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from typing import Callable, Union, List
from pathlib import Path

Time = float

# ================================================ #
# Default parameters                               #
# ================================================ #

# DATA and OUTPUT
DATA_DIR = "benchmark_data_dir"
RESULTS_FILE = "results.txt"

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
MOMEMTUMS = [0.9]  # TODO not implemented
DISCRIMINATIVE_LRS = [False]
ONE_CYCLE_POLICIES = [False]
FINE_TUNES = [True]
# TODO add precision (fp16, fp32)

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
MOMENTUM:          {MOMEMTUMS[0]} # TODO currently momentum is not implement
DISCRIMINATIVE_LR: {DISCRIMINATIVE_LRS[0]}
ONE_CYCLE_POLICY:  {ONE_CYCLE_POLICIES[0]}
FINE_TUNE:         {FINE_TUNES[0]}

Example usage:

# Test the effect of 3 learning rates on 3 batch sizes
$ python benchmark.py -l 1e-3 1e-4 1e-5 -bs 8 16 32 -o learning_rate_batch_size.txt

# Test the effect of one cycle policy without using discriminative learning rates
$ python benchmark.py -dl False -ocp True False

# Test different architectures and image sizes
$ python benchmark.py -a squeezenet1_1 resenet18 resnet50 -is 299 499
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
MOMENTUM:          {MOMEMTUM}
DISCRIMINATIVE_LR: {DISCRIMINATIVE_LR}
ONE_CYCLE_POLICY:  {ONE_CYCLE_POLICY}
FINE_TUNE:         {FINE_TUNE}
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

# Available architectures for this script
architecture_map = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "squeezenet1_1": models.squeezenet1_1,
}

# ================================================ #


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _get_parser():

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
        "--momentum",
        "-m",
        dest="momentums",
        nargs="+",
        help="momentums - recommended options: [0.9]",
        default=MOMEMTUMS,
        type=float,
    )
    parser.add_argument(
        "--discriminative-lr",
        "-dl",
        dest="discriminative_lrs",
        nargs="+",
        help="discriminative lr - options: [True, False]. To use discriminative learning rates, fine_tune must equal True",
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
    parser.add_argument(
        "--fine_tune",
        "-ft",
        dest="fine_tunes",
        nargs="+",
        help="fine tune (unfreeze all layers) - options: [True, False]",
        default=FINE_TUNES,
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
    for a in args.transforms:
        assert a in [True, False]
    for a in args.discriminative_lrs:
        assert a in [True, False]
    for a in args.one_cycle_policies:
        assert a in [True, False]
    for a in args.fine_tunes:
        assert a in [True, False]

    # if discriminative lr is on, we cannot have fine tune set to false
    if True in args.discriminative_lrs:
        assert False not in args.fine_tunes

    # get mapping of model object: ex. "resnet34" --> models.resnet34
    architecture_params = []
    for a in args.architectures:
        architecture_params.append(architecture_map[a])
    args.architectures = architecture_params

    return args


def _get_permutations(args):
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
        args.momentums,
        args.discriminative_lrs,
        args.one_cycle_policies,
        args.fine_tunes,
    ]
    permutations = list(itertools.product(*params))
    print(f"Trying {len(permutations)} permutations...\n")
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
    data: ImageDataBunch,
    arch: Callable,
    epoch: int,
    lr: float,
    p: float,
    wd: float,
    moms: float,
    discriminative_lr: bool,
    one_cycle_policy: bool,
    fine_tune: bool,
    stop_early: bool,
) -> Tuple[Learner, Time]:
    """
    """
    start = time.time()

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

    # fine tune & discriminative lr
    if fine_tune:
        learn.unfreeze()
        if discriminative_lr:
            lr = slice(lr / 10, lr / 5, lr)
    else:
        if discriminative_lr:
            raise Exception(
                "Cannot run discriminative_lr when fine_tune is False"
            )

    # one cycle policy
    if one_cycle_policy:
        learn.fit_one_cycle(
            cyc_len=epoch, max_lr=lr, wd=wd  # , moms=moms TODO
        )
    else:
        learn.fit(epochs=epoch, lr=lr, wd=wd)  # , moms=moms TODO

    end = time.time()
    duration = end - start

    return learn, duration


def _download_datasets() -> List[Path]:
    """ Download all datasets to DATA_DIR """
    start = time.time()

    # make data dir if not exist
    if not Path(DATA_DIR).is_dir():
        os.makedirs(DATA_DIR)

    # download all data urls
    paths = list()
    for url in Urls.all():
        paths.append(unzip_url(url, DATA_DIR, exist_ok=True))

    end = time.time()
    print(f"Time to download and prepare data: {end-start}")

    return paths


if __name__ == "__main__":

    start = time.time()
    args = _get_parser()
    results_file = open(args.output, "w")
    paths = _download_datasets()
    permutations = _get_permutations(args)

    for r in range(args.repeat):
        for i, p in enumerate(permutations):
            print(
                f"Running {i+1} of {len(permutations)} permutations. Repeat {r+1} of {args.repeat}."
            )

            LR, EPOCH, BATCH_SIZE, IM_SIZE, ARCHITECTURE, TRANSFORM, DROPOUT, WEIGHT_DECAY, MOMEMTUM, DISCRIMINATIVE_LR, ONE_CYCLE_POLICY, FINE_TUNE = (
                p
            )

            results_file.write(param_msg())

            results = dict()
            for path in paths:
                torch.cuda.empty_cache()
                data = _get_data_bunch(path, TRANSFORM, IM_SIZE, BATCH_SIZE)

                learn, duration = _learn(
                    data,
                    ARCHITECTURE,
                    EPOCH,
                    LR,
                    DROPOUT,
                    WEIGHT_DECAY,
                    MOMEMTUM,
                    DISCRIMINATIVE_LR,
                    ONE_CYCLE_POLICY,
                    FINE_TUNE,
                    args.early_stopping,
                )

                _, metric = learn.validate(
                    learn.data.valid_dl, metrics=[accuracy]
                )
                results[path] = {"duration": duration, "accuracy": metric}

            for path, val in results.items():
                results_file.write(result_msg())

    shutil.rmtree(DATA_DIR)
    end = time.time()
    results_file.write(time_msg())
    results_file.close()
