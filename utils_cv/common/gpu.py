# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import platform

import torch
from torch.cuda import current_device, get_device_name, is_available


def which_processor():
    """Check if torch is using GPU or CPU"""
    if is_available():
        device_nr = current_device()
        print(f"Torch is using GPU: {get_device_name(device_nr)}")
    else:
        print("Cuda is not available. Torch is using CPU")


def is_linux():
    """Returns if machine is running an Linux OS"""
    return platform.system().lower() == "linux"


def is_windows():
    """Returns if machine is running an Windows OS"""
    return platform.system().lower() == "windows"


def linux_with_gpu():
    """Returns if machine is running an Linux OS and has a GPU"""
    has_gpu = is_available()
    return is_linux() and has_gpu


def torch_device():
    """ Gets the torch device. Try gpu first, otherwise gpu. """
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def db_num_workers(non_windows_num_workers: int = 16):
    """Returns how many workers to use when loading images in a databunch. On windows machines using >0 works significantly slows down model
    training and evaluation. Setting num_workers to zero on Windows machines will speed up training/inference significantly, but will still be
    2-3 times slower.

    For a description of the slow windows speed see: https://github.com/pytorch/pytorch/issues/12831
    """
    if is_windows():
        return 0
    else:
        return non_windows_num_workers
