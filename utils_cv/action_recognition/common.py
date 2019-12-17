# Copyright (c) Microsoft
# Licensed under the MIT License.

import sys

import torch
import torch.cuda as cuda
import torchvision


class Config(object):
    def __init__(self, config=None, **extras):
        """Dictionary wrapper to access keys as attributes.

        Args:
            config (dict or Config): Configurations
            extras (kwargs): Extra configurations

        Examples:
            >>> cfg = Config({'lr': 0.01}, momentum=0.95)
            or
            >>> cfg = Config({'lr': 0.01, 'momentum': 0.95})
            then, use as follows:
            >>> print(cfg.lr, cfg.momentum)
        """
        if config is not None:
            if isinstance(config, dict):
                for k in config:
                    setattr(self, k, config[k])
            elif isinstance(config, self.__class__):
                self.__dict__ = config.__dict__.copy()
            else:
                raise ValueError("Unknown config")

        for k, v in extras.items():
            setattr(self, k, v)

    def get(self, key, default):
        return getattr(self, key, default)


def system_info():
    print(sys.version, "\n")
    print("PyTorch {}".format(torch.__version__), "\n")
    print("Torch-vision {}".format(torchvision.__version__), "\n")
    print("Available devices:")
    if cuda.is_available():
        for i in range(cuda.device_count()):
            print("{}: {}".format(i, cuda.get_device_name(i)))
    else:
        print("CPUs")
