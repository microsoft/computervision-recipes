from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
import os
import logging
import time
from logging.handlers import RotatingFileHandler
import itertools
import argparse
import torch

architecture_map = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

# DATA
DATA_DIR = "../data"
DATASET = [
    "fashionTexture",
    "fridge_objects",
    "recycle_v3",
    "food101Subset",
    "flickrLogos32Subset",
    "lettuce",
]
RESULTS_FILE = "results.txt"

# DEFAULT HYPERPARAMS TO EXPLORE:
LRS = [1e-4]
EPOCHS = [5]
BATCH_SIZES = [16]
IM_SIZES = [299, 499]
ARCHITECTURES = ["resnet50", "resnet18"]
TRANSFORMS = [True, False]


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
        "--output",
        "-o",
        dest="output",
        help="the name of the output file",
        default=RESULTS_FILE,
    )
    args = parser.parse_args()

    # check all input values are valid
    for a in args.architectures:
        assert a in ["resnet18", "resnet34", "resnet50"]
    for a in args.transforms:
        assert a in [True, False]

    # get mapping of model object "resnet34" --> models.resnet34
    architecture_params = []
    for a in args.architectures:
        architecture_params.append(architecture_map[a])
    args.architectures = architecture_params

    return args


def _get_permutations(args):
    l = [
        args.lrs,
        args.epochs,
        args.batch_sizes,
        args.im_sizes,
        args.architectures,
        args.transforms,
    ]
    return list(itertools.product(*l))


if __name__ == "__main__":

    # get args
    args = _get_parser()

    # setup results file
    results_file = open(args.output, "w")

    # get all permutations to test
    permutations = _get_permutations(args)

    results_file.write(f"Trying {len(permutations)} permutations...\n")
    for i, p in enumerate(permutations):
        print(f'Running {i} of {len(permutations)} permutations.')
        LR, EPOCH, BATCH_SIZE, IM_SIZE, ARCHITECTURE, TRANSFORM = p
        results_file.write(
            f"""
----------------------------------
RUN # {i+1} of {len(permutations)}
----------------------------------
LR:           {LR}
EPOCH:        {EPOCH}
IM_SIZE:      {IM_SIZE}
BATCH_SIZE:   {BATCH_SIZE}
ARCHITECTURE: {ARCHITECTURE}
TRANSFORMS:   {TRANSFORM}\n"""
        )

        results = dict()
        for data_dir in DATASET:

            # clear gpu mem
            torch.cuda.empty_cache()

            # create databunch
            path = Path(f"{DATA_DIR}/{data_dir}")
            tfms = get_transforms() if TRANSFORM else None
            data = (
                ImageItemList.from_folder(path)
                .random_split_by_pct(valid_pct=0.33, seed=10)
                .label_from_folder()
                .transform(tfms=tfms, size=IM_SIZE)
                .databunch(bs=BATCH_SIZE)
                .normalize(imagenet_stats)
            )

            # train
            learn = create_cnn(data, ARCHITECTURE, metrics=accuracy)
            learn.unfreeze()
            start = time.time()
            learn.fit(EPOCH, LR)
            end = time.time()
            duration = end - start

            # metrics
            _, metric = learn.validate(learn.data.valid_dl, metrics=[accuracy])
            results[data_dir] = {"duration": duration, "accuracy": metric}

        for data_dir, val in results.items():
            results_file.write(
                f"""
{data_dir}:
    duration: {val["duration"]}
    accuracy: {val["accuracy"]}\n"""
            )

    results_file.close()
