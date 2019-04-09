# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import warnings

from torch.cuda import current_device, get_device_name, is_available


def gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary {device_name, total_memory, used_memory}.
              Returns an empty list if there is no cuda device available.
    """
    gpus = []

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        for o in output.split("\n"):
            info = o.split(",")
            if len(info) == 3:
                gpu = dict()
                gpu["device_name"] = info[0].strip()
                gpu["total_memory"] = info[1].strip()
                gpu["used_memory"] = info[2].strip()
                gpus.append(gpu)
    except subprocess.CalledProcessError as e:
        warnings.warn(e.stdout)
    except FileNotFoundError:
        warnings.warn("GPU info is not available.")

    return gpus


def which_processor():
    """Check if fastai/torch is using GPU or CPU"""
    if is_available():
        print(f"Fast.ai (Torch) is using GPU: {get_device_name(0)}")
        gpu = gpu_info()[current_device()]
        free = int(gpu['total_memory']) - int(gpu['used_memory'])
        print(f"Available / Total memory = {free} / {gpu['total_memory']} (MiB)")
    else:
        print("Cuda is not available. Fast.ai/Torch is using CPU")
