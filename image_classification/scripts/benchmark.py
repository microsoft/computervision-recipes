import sys
sys.path.append("../")

import os
import logging
import time
import itertools
import argparse
import torch

from utils_ic.datasets import unzip_url, Urls
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import error_rate
from typing import Callable, Union
from pathlib import Path

# Type declaration(s)
Time = float

# Available architectures for this script
architecture_map = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "squeezenet1_1": models.squeezenet1_1
}

# DATA
DATA_DIR = "../data"
RESULTS_FILE = "results.txt"

# ENABLE EARLY STOPPING
EARLY_STOPPING = True

# DEFAULT HYPERPARAMS TO EXPLORE:
LRS = [1e-4]
EPOCHS = [5]
BATCH_SIZES = [16]
IM_SIZES = [299]
ARCHITECTURES = ["resnet50", "resnet18", "squeezenet1_1"]
TRANSFORMS = [True]
DROPOUTS = [0.5]
WEIGHT_DECAYS = [0.01]
MOMEMTUMS = [0.9]
EARLY_STOPPING = True
# TODO add precision (fp16, fp32)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr",
        "--lrs",
        "-l",
        dest="lrs",
        nargs="+",
        help="learning rate - recommended options: [1e-3, 1e-4, 1e-5] ",
        default=LRS,
        type=float,
    )
    parser.add_argument(
        "--epoch",
        "--epochs",
        "-e",
        dest="epochs",
        nargs="+",
        help="epochs - recommended options: [3, 5, 10]",
        default=EPOCHS,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "--batch-sizes",
        "-bs",
        dest="batch_sizes",
        nargs="+",
        help="batch sizes - recommended options: [8, 16, 32, 64]",
        default=BATCH_SIZES,
        type=int,
    )
    parser.add_argument(
        "--im-size",
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
        "--architectures",
        "-a",
        dest="architectures",
        nargs="+",
        help="architecture - options: ['resnet18', 'resnet34', 'resnet50']",
        default=ARCHITECTURES,
        type=str,
    )
    parser.add_argument(
        "--transform",
        "--tranforms",
        "-t",
        dest="transforms",
        nargs="+",
        help="tranform - options: [True, False]",
        default=TRANSFORMS,
        type=bool,
    )
    parser.add_argument(
        "--dropout",
        "--dropouts",
        "-d",
        dest="dropouts",
        nargs="+",
        help="dropout - recommended options: [0.5]",
        default=DROPOUTS,
        type=float,
    )
    parser.add_argument(
        "--weight-decay",
        "--weight-decays",
        "-wd",
        dest="weight_decays",
        nargs="+",
        help="weight decay - recommended options: [0.01]",
        default=WEIGHT_DECAYS,
        type=float,
    )
    parser.add_argument(
        "--momentum",
        "--momentums",
        "-m",
        dest="momentums",
        nargs="+",
        help="momentums - recommended options: [0.9]",
        default=MOMEMTUMS,
        type=float,
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
    l = [
        args.lrs,
        args.epochs,
        args.batch_sizes,
        args.im_sizes,
        args.architectures,
        args.transforms,
        args.dropouts,
        args.weight_decays,
        args.momentums,
    ]
    return list(itertools.product(*l))


def _get_data_bunch(
    path: Union[Path, str], transform: bool, im_size: int, bs: int
) -> ImageDataBunch:
    """
    """
    path = path if type(path) is Path else Path(path)
    tfms = get_transforms() if transform else None
    return (
        ImageItemList.from_folder(path)
        .random_split_by_pct(valid_pct=0.33, seed=10)
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
    stop_early: bool,
) -> Tuple[Learner, Time]:
    """
    """

    # callbacks to pass to learner
    callbacks = list()
    if stop_early:
        callbacks.append(
            partial(
                EarlyStoppingCallback,
                monitor="accuracy",
                min_delta=0.01,
                patience=3,
            )
        )

    # create learner
    learn = create_cnn(
        data, arch, metrics=accuracy, ps=p, callback_fns=callbacks
    )
    learn.unfreeze()
    start = time.time()
    learn.fit(
        epochs=epoch, lr=lr , wd=wd #, moms=moms TODO
    )
    end = time.time()
    duration = end - start

    return learn, duration


if __name__ == "__main__":

    # get args
    args = _get_parser()

    # setup results file
    results_file = open(args.output, "w")

    # get all permutations to test
    permutations = _get_permutations(args)
    results_file.write(f"Trying {len(permutations)} permutations...\n")

    # download data if not exist
    start = time.time()
    paths = list()
    for url in Urls.all():
        paths.append(unzip_url(url, exist_ok=True))
    end = time.time()
    print(f"Time to download and prepare data: {end-start}")

    # iterate through each permutation
    for i, p in enumerate(permutations):
        print(f"Running {i} of {len(permutations)} permutations.")

        LR, EPOCH, BATCH_SIZE, IM_SIZE, ARCHITECTURE, TRANSFORM, DROPOUT, \
        WEIGHT_DECAY, MOMEMTUM = p

        message = (
            f"----------------------------------\n"
            f"RUN # {i+1} of {len(permutations)}\n"
            f"----------------------------------\n"
            f"LR:           {LR}                \n"
            f"EPOCH:        {EPOCH}             \n"
            f"IM_SIZE:      {IM_SIZE}           \n"
            f"BATCH_SIZE:   {BATCH_SIZE}        \n"
            f"ARCHITECTURE: {ARCHITECTURE}      \n"
            f"TRANSFORMS:   {TRANSFORM}         \n"
            f"DROPOUT:      {DROPOUT}           \n"
            f"WEIGHT_DECAY: {WEIGHT_DECAY}      \n"
            f"MOMENTUM:     {MOMEMTUM}          \n"
        )
        results_file.write(message)

        results = dict()
        # run training for each dataset
        for path in paths:

            # clear gpu mem
            torch.cuda.empty_cache()

            # create databunch
            data = _get_data_bunch(
                path, TRANSFORM, IM_SIZE, BATCH_SIZE
            )

            # train
            learn, duration = _learn(
                data,
                ARCHITECTURE,
                EPOCH,
                LR,
                DROPOUT,
                WEIGHT_DECAY,
                MOMEMTUM,
                args.early_stopping,
            )

            # metrics
            _, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
            results[path] = {"duration": duration, "accuracy": metric}

        # write results to file
        for path, val in results.items():
            message = (
                f"{os.path.basename(path)}:\n"
                f"\tduration: {val['duration']}\n"
                f"\taccuracy: {val['accuracy']}\n"
            )
            results_file.write(message)

    results_file.close()
