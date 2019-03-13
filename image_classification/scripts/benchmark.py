import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import shutil
from utils_ic.benchmark import TrainingSchedule, Architecture, benchmark
from utils_ic.datasets import unzip_urls, data_path
from argparse import RawTextHelpFormatter, Namespace
from pathlib import Path

# DATA and OUTPUT
DATA_DIR = Path(data_path()) / "tmp_data"
RESULTS_FILE = Path(data_path()) / "results.csv"

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
# TODO add precision (fp16, fp32), add matasetsomentum

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

time_msg = lambda: f"""Total Time elapsed: {round(end - start, 1)} seconds."""

output_msg = (
    lambda: f"""Output has been saved to '{os.path.realpath(args.output)}'."""
)


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
    parser.add_argument(
        "--inputs",
        "-i",
        dest="inputs",
        nargs="+",
        help="A list of data paths to run the tests on. The datasets must be structured so that each class is in a separate folder.",
        default=unzip_urls(DATA_DIR),
        type=str,
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


if __name__ == "__main__":

    start = time.time()

    args = _get_parser()

    params = (
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
    )

    datasets = args.inputs
    repeat = args.repeat
    early_stopping = args.early_stopping

    results_df = benchmark(datasets, params, repeat, early_stopping)
    results_df.to_csv(args.output)

    shutil.rmtree(DATA_DIR)

    end = time.time()
    print(time_msg())
    print(output_msg())
