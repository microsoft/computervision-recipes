# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import warnings


def gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary {device_name, total_memory, used_memory}.
              Returns an empty list if there is no cuda device available.
    """
    gpus = []

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        )
        for o in output.split('\n'):
            info = o.split(',')
            if len(info) == 3:
                gpu = dict()
                gpu['device_name'] = info[0].strip()
                gpu['total_memory'] = info[1].strip()
                gpu['used_memory'] = info[2].strip()
                gpus.append(gpu)
    except subprocess.CalledProcessError as e:
        warnings.warn(e.stdout)
    except FileNotFoundError:
        warnings.warn("nvidia-smi is not available.")

    return gpus
