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


def linux_with_gpu():
    """Returns if machine is running an Linux OS and has a GPU"""
    is_linux = platform.system().lower() == "linux"
    has_gpu = is_available()
    return is_linux and has_gpu


def torch_device():
    """ Gets the torch device. Try gpu first, otherwise gpu. """
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
