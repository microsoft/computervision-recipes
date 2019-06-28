# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch.cuda import current_device, get_device_name, is_available


def which_processor():
    """Check if fastai/torch is using GPU or CPU"""
    if is_available():
        device_nr = current_device()
        print(f"Fast.ai (Torch) is using GPU: {get_device_name(device_nr)}")
    else:
        print("Cuda is not available. Fast.ai/Torch is using CPU")


# Helper function to check if GPU machine which linux.
# Integration tests are too slow on Windows/CPU machines.
def linux_with_gpu():
    """Returns if machine is running an Linux OS and has a GPU"""
    is_linux = platform.system().lower() == "linux"
    has_gpu = is_available()
    return is_linux and has_gpu
