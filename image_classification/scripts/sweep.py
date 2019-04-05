import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import shutil

from utils_ic.parameter_sweeper import *
from utils_ic.datasets import data_path
from argparse import RawTextHelpFormatter, Namespace
from pathlib import Path

argparse_desc_msg = """
This script is used to benchmark the different hyperparameters when it comes to doing image classification.

This script will run all permutations of the parameters that are passed in.

This script will either run these tests on:
- an input dataset defined by --input
- a set of benchmarking datasets defined by --benchmark, which will create a
  temporary data directory with all benchmarking datasets loaded into it, and delete it at the end.

This script uses accuracy as the evaluation metric.

Use [-W ignore] to ignore warning messages when running the script.
"""

argparse_epilog_msg = """
Example usage:
{default_params}

# Test the effect of 3 learning rates on 3 batch sizes
$ python sweep.py -lr 1e-3 1e-4 1e-5 -bs 8 16 32 -i <input_data> -o learning_rate_batch_size.csv

# Test the effect of one cycle policy without using discriminative learning rates over 5 runs
$ python sweep.py -dl False -ocp True False -r 5 -i <input_data> -o ocp_dl.csv

# Test different architectures and image sizes
$ python sweep.py -a squeezenet1_1 resenet18 resnet50 -is 299 499 -i <input_data> -o arch_im_sizes.csv

# Test different training schedules over 3 runs on the benchmark dataset
$ python sweep.py -ts body_only head_first_then_body -r 3 --benchmark -o training_schedule.csv

---

To view results, we recommend using pandas dataframes:

```
import pandas as pd
df = pd.read_csv("results.csv", index_col=[0, 1, 2])
```

""".format

time_msg = """Total Time elapsed: {time} seconds.""".format

output_msg = """Output has been saved to '{output_path}'.""".format


def _str_to_bool(string: str) -> bool:
    """ Convert string to bool. """
    if string.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif string.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _get_parser(default_params: Dict[str, List[Any]]) -> Namespace:
    """ Get parser for this script. """
    parser = argparse.ArgumentParser(
        description=argparse_desc_msg,
        epilog=argparse_epilog_msg(default_params=default_params),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        dest="learning_rates",
        nargs="+",
        help="Learning rate - recommended options: [1e-3, 1e-4, 1e-5] ",
        type=float,
    )
    parser.add_argument(
        "--epoch",
        "-e",
        dest="epochs",
        nargs="+",
        help="Epochs - recommended options: [3, 5, 10, 15]",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        dest="batch_sizes",
        nargs="+",
        help="Batch sizes - recommended options: [8, 16, 32, 64]",
        type=int,
    )
    parser.add_argument(
        "--im-size",
        "-is",
        dest="im_sizes",
        nargs="+",
        help="Image sizes - recommended options: [299, 499]",
        type=int,
    )
    parser.add_argument(
        "--architecture",
        "-a",
        dest="architectures",
        nargs="+",
        choices=["squeezenet1_1", "resnet18", "resnet34", "resnet50"],
        help="Choose an architecture.",
        type=str,
    )
    parser.add_argument(
        "--transform",
        "-t",
        dest="transforms",
        nargs="+",
        help="Tranform (data augmentation) - options: [True, False]",
        type=_str_to_bool,
    )
    parser.add_argument(
        "--dropout",
        "-d",
        dest="dropouts",
        nargs="+",
        help="Dropout - recommended options: [0.5]",
        type=float,
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        dest="weight_decays",
        nargs="+",
        help="Weight decay - recommended options: [0.01]",
        type=float,
    )
    parser.add_argument(
        "--training-schedule",
        "-ts",
        dest="training_schedules",
        nargs="+",
        choices=["head_only", "body_only", "head_first_then_body"],
        help="Choose a training schedule",
        type=str,
    )
    parser.add_argument(
        "--discriminative-lr",
        "-dl",
        dest="discriminative_lrs",
        nargs="+",
        help="Discriminative learning rate - options: [True, False]. To use discriminative learning rates, training schedule must not be 'head_only'",
        choices=["True", "False"],
        type=_str_to_bool,
    )
    parser.add_argument(
        "--one-cycle-policy",
        "-ocp",
        dest="one_cycle_policies",
        nargs="+",
        help="one cycle policy - options: [True, False]",
        type=_str_to_bool,
    )
    i_parser = parser.add_mutually_exclusive_group(required=True)
    i_parser.add_argument(
        "--inputs",
        "-i",
        dest="inputs",
        nargs="+",
        help="A list of data paths to run the tests on. The datasets must be structured so that each class is in a separate folder. <--benchmark> must be False",
        type=str,
    )
    i_parser.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="Whether or not to use curated benchmark datasets to test. <--input> must be empty",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Stop training early if possible",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        dest="repeat",
        help="The number of times to repeat each permutation",
        type=int,
    )
    parser.add_argument(
        "--output", "-o", dest="output", help="The path of the output file."
    )
    parser.set_defaults(
        repeat=3, early_stopping=False, inputs=None, benchmark=False
    )
    args = parser.parse_args()

    # if discriminative lr is on, we cannot have a 'head_only'
    # training_schedule
    if args.discriminative_lrs is not None and True in args.discriminative_lrs:
        assert "head_only" not in args.training_schedules

    # get mapping of architecture enum: ex. "resnet34" -->
    # Architecture.resnet34 -> models.resnet34
    if args.architectures is not None:
        args.architectures = [Architecture[a] for a in args.architectures]

    # get mapping of training enum: ex. "head_only" -->
    # TrainingSchedule.head_only --> 0
    if args.training_schedules is not None:
        args.training_schedules = [
            TrainingSchedule[t] for t in args.training_schedules
        ]

    return args


if __name__ == "__main__":

    start = time.time()
    sweeper = ParameterSweeper()
    args = _get_parser(sweeper.parameters)

    sweeper.update_parameters(
        learning_rate=args.learning_rates,
        epochs=args.epochs,
        batch_size=args.batch_sizes,
        im_size=args.im_sizes,
        architecture=args.architectures,
        transform=args.transforms,
        dropout=args.dropouts,
        weight_decay=args.weight_decays,
        training_schedule=args.training_schedules,
        discriminative_lr=args.discriminative_lrs,
        one_cycle_policy=args.one_cycle_policies,
    )

    data = args.inputs
    if not data:
        data = ParameterSweeper.download_benchmark_datasets(
            Path(data_path()) / "benchmark_data"
        )

    df = sweeper.run(
        datasets=data, reps=args.repeat, early_stopping=args.early_stopping
    )
    df.to_csv(args.output)

    if args.benchmark:
        for path in args.inputs:
            shutil.rmtree(path)

    end = time.time()
    print(time_msg(time=round(end - start, 1)))
    print(output_msg(output_path=os.path.realpath(args.output)))
