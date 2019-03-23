import subprocess


def gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary {device_name, total_memory, used_memory}.
    """
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,nounits,noheader"],
        encoding='utf-8'
    )

    gpus = []
    for o in output.split('\n'):
        info = o.split(',')
        if len(info) == 3:
            gpu = dict()
            gpu['device_name'] = info[0].strip()
            gpu['total_memory'] = info[1].strip()
            gpu['used_memory'] = info[2].strip()
            gpus.append(gpu)

    return gpus
